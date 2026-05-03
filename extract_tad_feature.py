"""Extract features using VideoMamba model"""
import argparse
import os
import sys

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms
import decord
from decord import VideoReader

# Add videomamba to path and import models to register them with timm
videomamba_path = os.path.join(os.path.dirname(__file__), 'videomamba', 'video_sm')
if videomamba_path not in sys.path:
    sys.path.insert(0, videomamba_path)
# Import models to register them with timm's create_model
from models import videomamba_tiny, videomamba_small, videomamba_middle  # noqa: F401

decord.bridge.set_bridge("torch")


def to_normalized_float_tensor(vid):
    """Convert video from (T, H, W, C) uint8 to (C, T, H, W) float32 normalized to [0, 1]"""
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255.0


def resize(vid, size, interpolation='bilinear'):
    """Resize video tensor"""
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


# Common video extensions
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv')


def get_args():
    parser = argparse.ArgumentParser(
        'Extract features using VideoMamba model', add_help=False)

    parser.add_argument(
        '--video_path',
        default=None,
        type=str,
        help='path to a single video file (optional)')
    
    parser.add_argument(
        '--data_path',
        default=['FootballAQA/Premier League/4th', 'FootballAQA/Premier League/5th'],
        nargs='+',
        type=str,
        help='one or more folders containing videos (used when video_path is not set)')
    
    parser.add_argument(
        '--save_path',
        default='./features',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='videomamba_middle',
        choices=['videomamba_tiny', 'videomamba_small', 'videomamba_middle'],
        type=str,
        metavar='MODEL',
        help='Name of VideoMamba model')
    
    parser.add_argument(
        '--ckpt_path',
        default=None,
        type=str,
        help='path to checkpoint file (optional, will use pretrained if not provided)')
    
    parser.add_argument(
        '--num_frames',
        default=8,
        type=int,
        help='number of frames per clip')
    
    parser.add_argument(
        '--stride',
        default=4,
        type=int,
        help='stride for sliding window extraction')
    
    parser.add_argument(
        '--op',
        default='pool',
        choices=['pool', 'orig'],
        type=str,
        help='feature saving mode: pool (clip-level) or orig (raw forward_features output)')

    return parser.parse_args()


def _pool_feature_like_swint(feat: torch.Tensor) -> torch.Tensor:
    """
    Convert VideoMamba forward_features output into a clip-level feature vector
    similar to Swin pooled output shape [B, C].

    Common VideoMamba output examples:
      - [B, C, L] -> average over L
      - [B, L, C] -> average over L
      - [B, C]    -> keep as-is
    """
    if feat.ndim == 2:
        return feat

    if feat.ndim == 3:
        # Heuristic: channel dim is usually larger than token/time dim.
        # Example from your run: [B, 576, 68] -> pool dim=-1.
        if feat.shape[1] >= feat.shape[2]:
            return feat.mean(dim=-1)
        return feat.mean(dim=1)

    # Fallback for higher-rank outputs: average all non-batch, non-channel dims.
    # Assume dim=1 is channel for this fallback.
    reduce_dims = tuple(range(2, feat.ndim))
    return feat.mean(dim=reduce_dims)


def load_video_frames(vr, num_frames, start_idx):
    """Load a clip of frames from video starting at start_idx
    
    Args:
        vr: VideoReader instance
        num_frames: number of frames to load
        start_idx: starting frame index
    Returns:
        frames: (T, H, W, C) uint8 tensor
    """
    vlen = len(vr)
    
    # Get frame indices
    end_idx = min(start_idx + num_frames, vlen)
    frame_indices = list(range(start_idx, end_idx))
    
    # Pad with last frame if needed
    if len(frame_indices) < num_frames:
        frame_indices = frame_indices + [frame_indices[-1]] * (num_frames - len(frame_indices))
    
    # Load frames: returns (T, H, W, C) uint8 tensor
    frames = vr.get_batch(frame_indices)
    return frames


def get_video_list(args):
    """Return list of (video_path, video_name) to process. video_name is used for the .npy filename."""
    if args.video_path is not None:
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        name = os.path.splitext(os.path.basename(args.video_path))[0]
        return [(args.video_path, name)]
    # Folder mode: scan one or more data_path entries for video files
    data_paths = args.data_path if isinstance(args.data_path, list) else [args.data_path]
    out = []
    seen_names = {}
    for data_path in data_paths:
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Data path is not a directory: {data_path}")
        for f in sorted(os.listdir(data_path)):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                video_path = os.path.join(data_path, f)
                name = os.path.splitext(f)[0]
                if name in seen_names:
                    print(
                        f"Warning: duplicate video name '{name}' in "
                        f"{seen_names[name]} and {video_path}. "
                        "Features may overwrite each other."
                    )
                seen_names[name] = video_path
                out.append((video_path, name))
    return out


def extract_feature_for_video(video_path, video_name, save_path, model, transform, num_frames, stride, op):
    """Extract features for one video and save as {video_name}.npy in save_path."""
    npy_path = os.path.join(save_path, f"{video_name}.npy")
    print(f"Loading video: {video_path}")
    vr = VideoReader(video_path, num_threads=1)
    vlen = len(vr)
    print(f"  Frames: {vlen}, clips (stride={stride}): {len(range(0, vlen - num_frames + 1, stride))}")

    feature_list = []
    num_clips = 0
    for start_idx in range(0, vlen - num_frames + 1, stride):
        frames = load_video_frames(vr, num_frames, start_idx)
        frames_transformed = transform(frames)
        input_data = frames_transformed.unsqueeze(0).cuda()
        with torch.no_grad():
            feature = model.forward_features(input_data)
            if op == 'pool':
                feature = _pool_feature_like_swint(feature)
            feature_list.append(feature.cpu().numpy())
        num_clips += 1
        if num_clips % 10 == 0:
            print(f"  Processed {num_clips} clips...")

    features = np.vstack(feature_list)
    np.save(npy_path, features)
    print(f"  Saved {npy_path} shape={features.shape} (op={op})")
    return npy_path


def extract_feature(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    transform = transforms.Compose([
        ToFloatTensorInZeroOne(),
        Resize((224, 224))
    ])

    video_list = get_video_list(args)
    if not video_list:
        print("No videos to process.")
        return
    print(f"Found {len(video_list)} video(s) to consider")

    num_frames = args.num_frames
    num_classes = 1000

    if args.ckpt_path is not None:
        print(f"Loading checkpoint from: {args.ckpt_path}")
        raw_ckpt = torch.load(args.ckpt_path, map_location='cpu')
        ckpt = raw_ckpt
        if 'model' in ckpt:
            ckpt = ckpt['model']
        elif 'module' in ckpt:
            ckpt = ckpt['module']
        elif 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new_ckpt = {}
        for k, v in ckpt.items():
            new_k = k.replace('module.', '') if k.startswith('module.') else k
            new_ckpt[new_k] = v
        ckpt = new_ckpt

        if 'temporal_pos_embedding' in ckpt:
            t_dim = ckpt['temporal_pos_embedding'].shape[1]
            num_frames = t_dim
            print(f"Inferred num_frames={num_frames} from checkpoint")
        if 'head.weight' in ckpt:
            num_classes = ckpt['head.weight'].shape[0]
            print(f"Inferred num_classes={num_classes} from checkpoint")

    print(f"Creating model: {args.model} (num_frames={num_frames}, num_classes={num_classes})")
    model = create_model(
        args.model,
        img_size=224,
        pretrained=args.ckpt_path is None,
        num_classes=num_classes,
        num_frames=num_frames,
        drop_path_rate=0.1,
    )
    if args.ckpt_path is not None:
        model.load_state_dict(ckpt, strict=True)
        print("Checkpoint loaded successfully")
    model.eval()
    model.cuda()

    for idx, (video_path, video_name) in enumerate(video_list):
        npy_path = os.path.join(args.save_path, f"{video_name}.npy")
        if os.path.exists(npy_path):
            print(f"[{idx + 1}/{len(video_list)}] Skip (already exists): {video_name}.npy")
            continue
        print(f"[{idx + 1}/{len(video_list)}] Extracting: {video_name}")
        try:
            extract_feature_for_video(
                video_path, video_name, args.save_path,
                model, transform, num_frames, args.stride, args.op
            )
        except Exception as e:
            print(f"  Error: {e}")
            raise


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
