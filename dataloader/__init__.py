"""Paper submission snapshot: Football dataloader only."""


def build_dataloader(cfg):
    name = cfg.dataset_name
    if name == "football":
        from dataloader.dataloader_football import get_dataloader

        return get_dataloader(cfg)
    raise ValueError(f"Unsupported dataset name [{name}]")


def build_kfold_dataloaders(cfg):
    raise NotImplementedError(
        "K-fold dataloaders are not included in this release snapshot."
    )
