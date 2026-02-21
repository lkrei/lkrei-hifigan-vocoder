from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config, device):
    datasets = instantiate(config.datasets)

    dataloaders = {}
    for partition_name in config.datasets.keys():
        dataset = datasets[partition_name]

        assert config.dataloader.batch_size <= len(dataset), (
            f"Batch size ({config.dataloader.batch_size}) cannot be "
            f"larger than dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(partition_name == "train"),
            shuffle=(partition_name == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[partition_name] = partition_dataloader

    batch_transforms = {"train": None, "inference": None}
    return dataloaders, batch_transforms
