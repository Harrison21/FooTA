import os
import csv
import time
import torch
import numpy as np
import wandb
from tqdm import tqdm
import logging
from utils.utils import log_and_print
from utils.vis import *
from loss import attention_loss, cal_spearmanr_rl2
# from methods.weight_methods import NashMTL
import matplotlib.pyplot as plt


def _unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def _should_log_pool_to_wandb(cfg, split, epoch):
    if not bool(cfg.get("graph_pool_log_wandb", False)):
        return False
    target_split = str(cfg.get("graph_pool_log_split", "test"))
    every_n = max(int(cfg.get("graph_pool_log_every_n_epochs", 1)), 1)
    return split == target_split and (epoch % every_n == 0)


def _make_pool_heatmap(pool_info, cfg, epoch, split):
    weights = pool_info.get("weights")
    valid_counts = pool_info.get("valid_counts")
    if weights is None or valid_counts is None or weights.numel() == 0:
        return None

    max_windows = max(int(cfg.get("graph_pool_log_max_windows", 4)), 1)
    max_nodes = max(int(cfg.get("graph_pool_log_max_nodes", 32)), 1)
    num_heads = int(weights.shape[0])
    num_windows = min(int(weights.shape[1]), max_windows)
    max_valid = min(int(valid_counts[:num_windows].max().item()), max_nodes)
    if num_windows <= 0 or max_valid <= 0:
        return None

    fig, axes = plt.subplots(num_heads, 1, figsize=(max(8, max_valid * 0.35), max(2.5, num_heads * 2.2)), squeeze=False)
    axes = axes[:, 0]
    for head_idx in range(num_heads):
        head_weights = weights[head_idx, :num_windows, :max_valid].numpy()
        ax = axes[head_idx]
        im = ax.imshow(head_weights, aspect="auto", cmap="viridis")
        ax.set_ylabel(f"head {head_idx}")
        ax.set_yticks(range(num_windows))
        ax.set_yticklabels([f"win {i}" for i in range(num_windows)])
        if head_idx == num_heads - 1:
            ax.set_xlabel("node index")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"{split} epoch {epoch} graph pool weights")
    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def _summarize_pool_info(pool_info):
    weights = pool_info.get("weights")
    valid_counts = pool_info.get("valid_counts")
    if weights is None or valid_counts is None or weights.numel() == 0:
        return {}

    eps = 1e-8
    summary = {
        "num_windows": float(weights.shape[1]),
        "avg_valid_nodes": float(valid_counts.float().mean().item()),
    }
    for head_idx in range(weights.shape[0]):
        head_weights = weights[head_idx]
        entropy = -(head_weights * (head_weights.clamp_min(eps).log())).sum(dim=-1)
        summary[f"head_{head_idx}/max_weight"] = float(head_weights.max().item())
        summary[f"head_{head_idx}/mean_entropy"] = float(entropy.mean().item())
        summary[f"head_{head_idx}/mean_active_nodes"] = float((head_weights > 0.01).float().sum(dim=-1).mean().item())
    return summary


def run(cfg, base_logger, network, data_loaders, kld, mse, optimizer, scheduler, awl, splits=["train","test"], wandb_run=None):
    # Unpack network - trailing entries may be None (video-only / graph-only /
    # non-late-fusion modes). Length-8 tuple supports late fusion with an
    # independent video neck + head.
    video_neck = None
    video_head = None
    if len(network) == 8:
        (backbone, neck, head, graph_encoder, fusion, video_projector,
         video_neck, video_head) = network
    elif len(network) == 6:
        backbone, neck, head, graph_encoder, fusion, video_projector = network
    elif len(network) == 5:
        backbone, neck, head, graph_encoder, fusion = network
        video_projector = None
    else:
        backbone, neck, head = network[:3]
        graph_encoder, fusion, video_projector = None, None, None

    device = next(head.parameters()).device
    use_video = bool(cfg.get("use_video_branch", True)) and backbone is not None
    use_graph = bool(cfg.get("use_graph_branch", False)) and graph_encoder is not None

    # --- Graph-only prediction + video auxiliary loss --------------------
    # Final/inference output == graph output. Video branch runs through its
    # own independent neck+head and contributes only an auxiliary training
    # loss:   L_total = L_graph + lambda_aux * L_video.
    graph_aux_video = (
        bool(cfg.get("graph_aux_video", False))
        and use_video and use_graph
        and video_neck is not None and video_head is not None
    )
    lambda_aux = float(cfg.get("lambda_aux", 0.1))

    # --- (Legacy) Late output-level fusion -------------------------------
    # Graph and video branches run through independent neck+head stacks and
    # their final outputs are combined with a fixed weight. Disabled
    # automatically when graph_aux_video is on (no output ensemble).
    late_fusion = (
        (not graph_aux_video)
        and bool(cfg.get("late_fusion", False))
        and use_video and use_graph
        and video_neck is not None and video_head is not None
    )
    lambda_fuse = float(cfg.get("lambda_fuse", 0.05))

    # Regression (output_dim==1) vs logits/classification (output_dim>1):
    # pick the matching per-sample loss.
    output_dim = int(cfg.get("output_dim", 1))
    task_mode = "regression" if output_dim == 1 else "logits"

    def _task_loss(pred, target):
        """Task loss that works for both regression and classification."""
        if task_mode == "regression":
            # Match existing behavior: MSE on scalar / vector regression targets.
            return mse(pred, target)
        # Classification: pred is [B, C] logits, target is [B] class indices.
        if target.dtype != torch.long:
            target = target.long()
        return torch.nn.functional.cross_entropy(pred, target)

    if graph_aux_video:
        log_and_print(
            base_logger,
            f"[GraphAuxVideo] lambda_aux={lambda_aux:.4f}, task_mode={task_mode}, "
            f"final_output = graph_output; "
            f"L_total = L_graph + lambda_aux * L_video"
        )
    elif late_fusion:
        log_and_print(
            base_logger,
            f"[LateFusion] lambda_fuse={lambda_fuse:.4f}, task_mode={task_mode}, "
            f"formula: y = (1 - lambda) * y_graph + lambda * y_video"
        )

    logger_exp = logging.getLogger("experiment_logger")
    logger_exp.setLevel(logging.INFO)
    exp_handler = logging.FileHandler("experiment.log")
    exp_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    exp_handler.setFormatter(exp_formatter)
    logger_exp.addHandler(exp_handler)
    
    rho_best, epoch_best, rl2_best = 0, 0, 0
    test_srcc = []
    
    test_only = "train" not in splits
    if test_only:
        cfg.epoch_num = 400

    for epoch in range(cfg.epoch_num):
        for split in splits:
            pred_batches = []
            true_batches = []
            debug_rows = []
            epoch_pool_info = None
            
            if split == 'train':
                if backbone is not None:
                    backbone.train()
                head.train()
                neck.train()
                if graph_encoder is not None:
                    graph_encoder.train()
                if fusion is not None:
                    fusion.train()
                if video_projector is not None:
                    video_projector.train()
                if video_neck is not None:
                    video_neck.train()
                if video_head is not None:
                    video_head.train()
                torch.set_grad_enabled(True)
            else:
                if backbone is not None:
                    backbone.eval()
                head.eval()
                neck.eval()
                if graph_encoder is not None:
                    graph_encoder.eval()
                if fusion is not None:
                    fusion.eval()
                if video_projector is not None:
                    video_projector.eval()
                if video_neck is not None:
                    video_neck.eval()
                if video_head is not None:
                    video_head.eval()
                torch.set_grad_enabled(False)
                
            self_map_lst = []
            cross_map_lst = []
            losses = torch.zeros((), device=device)
            mse_sum = torch.zeros((), device=device)
            batch_count = 0
            data_time_sum = 0.0
            step_time_sum = 0.0
            data_wait_start = time.perf_counter()
            for data_ in data_loaders[split]:
                data_time_sum += time.perf_counter() - data_wait_start
                step_start = time.perf_counter()
                data, clip_info = data_
                score_key = "score" if "score" in data else "completeness"
                if score_key not in data:
                    raise KeyError(
                        "Batch does not contain a training target. Expected "
                        "'score' or 'completeness'."
                    )
                score = data[score_key].float().to(device, non_blocking=True)
                clip_feats = None
                if use_video:
                    video = data["video"]
                    if cfg.split_feats:
                        if "feats" in data:
                            clip_feats = data["feats"].to(device, non_blocking=True)
                        else:
                            bs, frame, feats  = video.shape
                            video = video.reshape(video.shape[0],3,48,16,224,224).to(device, non_blocking=True)
                            clip_feats = torch.empty(bs, video.shape[2], feats, device=device)
                            for i in range(frame):
                                clip_feats[:,i] = backbone(video[:,:,i,:,:,:])[1].squeeze(-1).squeeze(-1).squeeze(-1)
                    else:
                        bs, frame, h, w  = video.shape
                        video = video.to(device, non_blocking=True)
                        clip_feats = backbone(video)[1].squeeze(-1).squeeze(-1).permute(0,2,1)

                    # Explicit learnable projection (e.g. 576 -> 128) before fusion.
                    if clip_feats is not None and video_projector is not None:
                        clip_feats = video_projector(clip_feats)

                # -- graph branch ------------------------------------
                graph_feats = None
                if use_graph and "graph_nodes" in data:
                    g_nodes = data["graph_nodes"].to(device, non_blocking=True)       # [B, T, N, D_pos]
                    g_mask = data["graph_mask"].bool().to(device, non_blocking=True)   # [B, T, N]
                    g_fids = data["graph_frame_ids"].long().to(device, non_blocking=True)  # [B, T, N]
                    g_img = None
                    if "graph_image_feats" in data:
                        g_img = data["graph_image_feats"].to(device, non_blocking=True)  # [B, T, N, D_img]
                    graph_feats = graph_encoder(g_nodes, g_mask, g_fids, g_img)  # [B, T, Dg]
                    if epoch_pool_info is None and _should_log_pool_to_wandb(cfg, split, epoch):
                        epoch_pool_info = _unwrap_module(graph_encoder).consume_latest_pool_info()

                # -- combine video + graph ---------------------------
                # Three paths:
                #   (a) graph_aux_video: independent graph and video
                #       neck+head stacks. Final output = graph output.
                #       Video is used only for an auxiliary training loss.
                #   (b) late_fusion (legacy): output ensemble with fixed mix.
                #   (c) feature-level fusion (legacy): mix features, then
                #       run a single neck+head.
                probs_graph = None
                probs_video = None
                if graph_aux_video and graph_feats is not None and clip_feats is not None:
                    # Graph branch (sole predictor).
                    tgt_g, graph_attn = neck(graph_feats, train=False)
                    probs_g, weight_g, means_g, var_g = head(tgt_g)
                    # Video branch (auxiliary loss only). Separate neck+head,
                    # so gradients from video never flow into graph hidden
                    # features.
                    tgt_v, _video_attn = video_neck(clip_feats, train=False)
                    probs_v, _weight_v, _means_v, _var_v = video_head(tgt_v)

                    if task_mode == "regression":
                        if probs_g.ndim > 1 and probs_g.shape[-1] == 1:
                            probs_g = probs_g.squeeze(-1)
                        if probs_v.ndim > 1 and probs_v.shape[-1] == 1:
                            probs_v = probs_v.squeeze(-1)

                    # Graph-only final output: no ensemble, no fusion.
                    probs = probs_g
                    probs_graph = probs_g
                    probs_video = probs_v
                    weight, means, var = weight_g, means_g, var_g
                elif late_fusion and graph_feats is not None and clip_feats is not None:
                    # Graph branch (dominant): its own neck+head.
                    tgt_g, graph_attn = neck(graph_feats, train=False)
                    probs_g, weight_g, means_g, var_g = head(tgt_g)
                    # Video branch: independent neck+head. Gradients here do
                    # NOT flow back into graph hidden features.
                    tgt_v, _video_attn = video_neck(clip_feats, train=False)
                    probs_v, weight_v, means_v, var_v = video_head(tgt_v)

                    if probs_g.ndim > 1 and probs_g.shape[-1] == 1:
                        probs_g = probs_g.squeeze(-1)
                    if probs_v.ndim > 1 and probs_v.shape[-1] == 1:
                        probs_v = probs_v.squeeze(-1)

                    # Same linear-combination formula for regression (scalar
                    # targets) and logits (per-class scores). Lambda is a
                    # plain python float, so it is NOT learnable.
                    probs = (1.0 - lambda_fuse) * probs_g + lambda_fuse * probs_v

                    # Expose graph/video components for downstream debugging.
                    probs_graph = probs_g
                    probs_video = probs_v

                    # Use the graph neck's outputs for attention loss / head
                    # bookkeeping (graph is the dominant branch).
                    weight, means, var = weight_g, means_g, var_g
                else:
                    if graph_feats is not None:
                        if clip_feats is None:
                            clip_feats = graph_feats
                        else:
                            t_side = data.get("team_side_id")
                            if t_side is not None:
                                t_side = t_side.long().to(device, non_blocking=True)          # [B]
                            clip_feats = fusion(clip_feats, graph_feats, t_side)  # [B, T, D_fused]

                    if clip_feats is None:
                        raise RuntimeError("No active feature branch. Enable use_video_branch and/or use_graph_branch.")

                    tgt_weight, graph_attn = neck(clip_feats,train=False)#split=="train")
                    probs, weight, means, var = head(tgt_weight)

                # Keep score/prob shape aligned for both scalar and vector prediction.
                if probs.ndim > 1 and probs.shape[-1] == 1:
                    probs = probs.squeeze(-1)
                if score.ndim > 1 and score.shape[-1] == 1:
                    score = score.squeeze(-1)
                
                pred_detached = probs.detach()
                true_detached = score.detach()
                pred_batches.append(pred_detached.reshape(-1))
                true_batches.append(true_detached.reshape(-1))
                pred_np = pred_detached.float().cpu().numpy()
                true_np = true_detached.float().cpu().numpy()

                if split == "test" and cfg.get("debug_dump_test_preds", False):
                    if isinstance(clip_info, (list, tuple)):
                        clip_info_list = [str(x) for x in clip_info]
                    else:
                        clip_info_list = [str(clip_info)]

                    football_target_mode = cfg.get("football_target_mode", "pair")
                    football_scalar_mode = (
                        cfg.dataset_name == "football"
                        and (football_target_mode in {"avg", "home", "away", "diff"} or use_graph)
                    )

                    if football_scalar_mode:
                        pred_flat = pred_np.reshape(-1)
                        true_flat = true_np.reshape(-1)
                        bs = pred_flat.shape[0]
                        if len(clip_info_list) != bs:
                            clip_info_list = [str(clip_info)] * bs
                        for i in range(bs):
                            p = float(pred_flat[i])
                            t = float(true_flat[i])
                            label_name = "team" if use_graph else football_target_mode
                            debug_rows.append({
                                "clip_info": clip_info_list[i],
                                "pred_target": p,
                                "true_target": t,
                                "error_target": p - t,
                                "target_name": label_name,
                            })
                    # Football: one row per game with two team scores.
                    elif pred_np.ndim == 2 and true_np.ndim == 2 and pred_np.shape[1] == 2 and true_np.shape[1] == 2:
                        bs = pred_np.shape[0]
                        if len(clip_info_list) != bs:
                            clip_info_list = [str(clip_info)] * bs
                        for i in range(bs):
                            p0, p1 = float(pred_np[i, 0]), float(pred_np[i, 1])
                            t0, t1 = float(true_np[i, 0]), float(true_np[i, 1])
                            debug_rows.append({
                                "clip_info": clip_info_list[i],
                                "pred_team_0": p0,
                                "pred_team_1": p1,
                                "true_team_0": t0,
                                "true_team_1": t1,
                                "error_team_0": p0 - t0,
                                "error_team_1": p1 - t1,
                                "pred_avg": (p0 + p1) / 2.0,
                                "true_avg": (t0 + t1) / 2.0,
                                "pred_gap": abs(p0 - p1),
                                "true_gap": abs(t0 - t1),
                            })
                    else:
                        # Generic fallback: one scalar row per target.
                        pred_flat = pred_np.reshape(-1)
                        true_flat = true_np.reshape(-1)
                        for idx, (pred_v, true_v) in enumerate(zip(pred_flat, true_flat)):
                            item = clip_info_list[idx] if idx < len(clip_info_list) else f"{clip_info_list[0]}#target_{idx}"
                            debug_rows.append({
                                "clip_info": item,
                                "pred_team_0": float(pred_v),
                                "pred_team_1": "",
                                "true_team_0": float(true_v),
                                "true_team_1": "",
                                "error_team_0": float(pred_v - true_v),
                                "error_team_1": "",
                                "pred_avg": float(pred_v),
                                "true_avg": float(true_v),
                                "pred_gap": "",
                                "true_gap": "",
                            })
                
                kld_loss, self_map_lst, cross_map_lst = attention_loss(graph_attn, kld, self_map_lst, cross_map_lst)
                # self_map_vis(graph_attn)
                # if epoch > 5000 and test_only:
                #     user_study(means,weight,clip_info,cfg.dataset_name)
             
                
                if cfg.dino_loss and split == "train":
                    _dino_std = float(cfg.get("dino_noise_std", 0.05))
                    probs = probs + torch.randn_like(probs) * _dino_std
                    if graph_aux_video and probs_video is not None:
                        # Apply matching noise to the video prediction so
                        # both branches train under the same stochastic
                        # conditions.
                        probs_video = probs_video + torch.randn_like(probs_video) * _dino_std
                # Main task loss on the final (graph) output.
                mes_loss = _task_loss(probs, score)
                mse_sum = mse_sum + mes_loss.detach()
                batch_count += 1
                if cfg.att_loss:
                    loss = awl(mes_loss, kld_loss)
                    # loss = mes_loss + kld_loss
                else:
                    loss = mes_loss

                # --- auxiliary video loss (graph_aux_video mode) --------
                # Only applied during training. Does not affect evaluation
                # metrics (which are computed from `probs` == graph output).
                if graph_aux_video and split == "train" and probs_video is not None:
                    aux_video_loss = _task_loss(probs_video, score)
                    loss = loss + lambda_aux * aux_video_loss
          
                losses = losses + loss.detach()

                if split=="train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                step_time_sum += time.perf_counter() - step_start
                data_wait_start = time.perf_counter()
                
                    
            # show information
            if pred_batches:
                pred_scores = torch.cat(pred_batches, dim=0).float().cpu().numpy().tolist()
                true_scores = torch.cat(true_batches, dim=0).float().cpu().numpy().tolist()
            else:
                pred_scores = []
                true_scores = []
            # Compute metrics once per split using all samples of that split.
            if cfg.dataset_name == "finediving":
                dataset_obj = data_loaders[split].dataset
                orig_ds = dataset_obj.dataset if isinstance(dataset_obj, torch.utils.data.Subset) else dataset_obj
                min_val, max_val = orig_ds.comp_min, orig_ds.comp_max
                denorm_preds = [(x * (max_val - min_val) + min_val) for x in pred_scores]
                denorm_trues = [(x * (max_val - min_val) + min_val) for x in true_scores]
                rho, p, rl2 = cal_spearmanr_rl2(denorm_preds, denorm_trues)
            else:
                rho, p, rl2 = cal_spearmanr_rl2(pred_scores, true_scores)

            epoch_mse = (mse_sum / max(batch_count, 1)).item()
            epoch_loss = (losses / max(batch_count, 1)).item()
            avg_data_time = data_time_sum / max(batch_count, 1)
            avg_step_time = step_time_sum / max(batch_count, 1)
            log_and_print(
                base_logger,
                f'Epoch: {epoch}, {split} correlation: {rho}, RL2: {rl2}, '
                f'MSE: {epoch_mse}, Loss: {epoch_loss}, data_time: {avg_data_time:.4f}s, '
                f'step_time: {avg_step_time:.4f}s, best_corr: {rho_best}'
            )
            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch,
                    f"{split}/srcc": float(rho),
                    f"{split}/rl2": float(rl2),
                    f"{split}/mse": float(epoch_mse),
                    f"{split}/loss": float(epoch_loss),
                    f"{split}/avg_data_time": float(avg_data_time),
                    f"{split}/avg_step_time": float(avg_step_time),
                    "best/test_srcc": float(rho_best),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                if epoch_pool_info is not None:
                    pool_summary = _summarize_pool_info(epoch_pool_info)
                    for key, value in pool_summary.items():
                        log_payload[f"{split}/graph_pool/{key}"] = value
                    heatmap = _make_pool_heatmap(epoch_pool_info, cfg, epoch, split)
                    if heatmap is not None:
                        log_payload[f"{split}/graph_pool/weights_heatmap"] = heatmap
                wandb_run.log(log_payload)

            if split == "test":
                pred_arr = np.asarray(pred_scores, dtype=np.float32)
                true_arr = np.asarray(true_scores, dtype=np.float32)
                pred_std = float(pred_arr.std()) if pred_arr.size else 0.0
                true_std = float(true_arr.std()) if true_arr.size else 0.0
                pred_min = float(pred_arr.min()) if pred_arr.size else 0.0
                pred_max = float(pred_arr.max()) if pred_arr.size else 0.0
                unique_pred = int(np.unique(np.round(pred_arr, 4)).shape[0]) if pred_arr.size else 0
                log_and_print(
                    base_logger,
                    f"[Debug][test] pred_std={pred_std:.6f}, true_std={true_std:.6f}, "
                    f"pred_range=({pred_min:.6f},{pred_max:.6f}), unique_pred_1e4={unique_pred}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "epoch": epoch,
                        "test/pred_std": pred_std,
                        "test/true_std": true_std,
                        "test/pred_min": pred_min,
                        "test/pred_max": pred_max,
                        "test/unique_pred_1e4": unique_pred,
                    })

                dump_every = int(cfg.get("debug_dump_every_n_epochs", 1))
                should_dump = cfg.get("debug_dump_test_preds", False) and (epoch % max(dump_every, 1) == 0)
                if should_dump:
                    dump_dir = cfg.get("debug_dump_dir", "exp/debug_test_preds")
                    os.makedirs(dump_dir, exist_ok=True)
                    dump_path = os.path.join(dump_dir, f"epoch_{epoch:04d}.csv")
                    football_target_mode = cfg.get("football_target_mode", "pair")
                    football_scalar_mode = (
                        cfg.dataset_name == "football"
                        and (football_target_mode in {"avg", "home", "away", "diff"} or use_graph)
                    )
                    with open(dump_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if football_scalar_mode:
                            writer.writerow([
                                "clip_info",
                                f"pred_{football_target_mode}",
                                f"true_{football_target_mode}",
                                f"error_{football_target_mode}",
                            ])
                            for row in debug_rows:
                                writer.writerow([
                                    row["clip_info"],
                                    row["pred_target"],
                                    row["true_target"],
                                    row["error_target"],
                                ])
                        else:
                            writer.writerow([
                                "clip_info",
                                "pred_team_0", "pred_team_1",
                                "true_team_0", "true_team_1",
                                "error_team_0", "error_team_1",
                                "pred_avg", "true_avg",
                                "pred_gap", "true_gap",
                            ])
                            for row in debug_rows:
                                writer.writerow([
                                    row["clip_info"],
                                    row["pred_team_0"], row["pred_team_1"],
                                    row["true_team_0"], row["true_team_1"],
                                    row["error_team_0"], row["error_team_1"],
                                    row["pred_avg"], row["true_avg"],
                                    row["pred_gap"], row["true_gap"],
                                ])
                    log_and_print(base_logger, f"[Debug][test] dumped predictions to: {dump_path}")
            
            if rho > rho_best and split == "test":
                rho_best = rho
                epoch_best = epoch
                rl2_best = rl2
                log_and_print(base_logger, '-----New best found!-----')
                if wandb_run is not None:
                    wandb_run.summary["best/test_srcc"] = float(rho_best)
                    wandb_run.summary["best/test_rl2"] = float(rl2_best)
                    wandb_run.summary["best/epoch"] = int(epoch_best)
                if not test_only:
                    ckpt_dict = {
                        'epoch': epoch,
                        'neck': neck.state_dict(),
                        'head': head.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_best,
                    }
                    if backbone is not None:
                        ckpt_dict['backbone'] = backbone.state_dict()
                    if graph_encoder is not None:
                        ckpt_dict['graph_encoder'] = graph_encoder.state_dict()
                    if fusion is not None:
                        ckpt_dict['fusion'] = fusion.state_dict()
                    if video_projector is not None:
                        ckpt_dict['video_projector'] = video_projector.state_dict()
                    if video_neck is not None:
                        ckpt_dict['video_neck'] = video_neck.state_dict()
                    if video_head is not None:
                        ckpt_dict['video_head'] = video_head.state_dict()
                    ckpt_dir = str(cfg.get("ckpt_dir", "ckpts"))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_name = cfg.get(
                        "ckpt_name",
                        f"{cfg.backbone}_{cfg.dataset_name}_{cfg.seed}_{cfg.label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}_{cfg.num_layers}",
                    )
                    if not str(ckpt_name).endswith(".pt"):
                        ckpt_name = f"{ckpt_name}.pt"
                    torch.save(
                        ckpt_dict,
                        os.path.join(ckpt_dir, ckpt_name),
                    )
                

    if test_only:                
        logger_exp.info(f"test dataset: {cfg.dataset_name}, seed: {cfg.seed}, label: {cfg.label}, query_var: {cfg.query_var}, pe: {cfg.pe}, att_loss: {cfg.att_loss}, dino_loss:{cfg.dino_loss}, num_layers: {cfg.num_layers}, SRCC: {rho_best}, RL2: {rl2_best}")
    # logger_exp.info(f"dataset: {cfg.dataset_name}, seed: {cfg.seed}, label: {cfg.label}, query_var: {cfg.query_var}, pe: {cfg.pe}, att_loss: {cfg.att_loss}, dino_loss:{cfg.dino_loss}, num_layers: {cfg.num_layers}, SRCC: {rho_best}, RL2: {rl2_best}")
