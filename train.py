import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifigan")
def main(config):
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    generator = instantiate(config.model).to(device)
    discriminator = instantiate(config.discriminator).to(device)
    logger.info(generator)

    optimizer_g = instantiate(config.optimizer_g, params=generator.parameters())
    optimizer_d = instantiate(config.optimizer_d, params=discriminator.parameters())

    lr_scheduler_g = instantiate(config.lr_scheduler_g, optimizer=optimizer_g)
    lr_scheduler_d = instantiate(config.lr_scheduler_d, optimizer=optimizer_d)

    epoch_len = config.trainer.get("epoch_len")
    mel_spec_config = OmegaConf.to_container(config.mel_spectrogram)

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        skip_oom=config.trainer.get("skip_oom", True),
        mel_spec_config=mel_spec_config,
        lambda_fm=config.lambda_fm,
        lambda_mel=config.lambda_mel,
    )

    trainer.train()


if __name__ == "__main__":
    main()
