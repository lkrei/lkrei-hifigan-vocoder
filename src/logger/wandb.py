from datetime import datetime

import numpy as np


class WandBWriter:
    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        try:
            import wandb

            wandb.login()
            self.run_id = run_id

            wandb.init(
                project=project_name,
                entity=entity,
                config=project_config,
                name=run_name,
                resume="allow",
                id=self.run_id,
                mode=mode,
                save_code=kwargs.get("save_code", False),
            )
            self.wandb = wandb
        except ImportError:
            logger.warning("Install wandb via: pip install wandb")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log(
            {self._object_name(scalar_name): scalar}, step=self.step
        )

    def add_scalars(self, scalars):
        self.wandb.log(
            {self._object_name(k): v for k, v in scalars.items()}, step=self.step
        )

    def add_image(self, image_name, image):
        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image)}, step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log(
            {
                self._object_name(audio_name): self.wandb.Audio(
                    audio, sample_rate=sample_rate
                )
            },
            step=self.step,
        )

    def add_text(self, text_name, text):
        self.wandb.log(
            {self._object_name(text_name): self.wandb.Html(text)}, step=self.step
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        values_for_hist = values_for_hist.detach().cpu().numpy()
        np_hist = np.histogram(values_for_hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(values_for_hist, bins=512)
        hist = self.wandb.Histogram(np_histogram=np_hist)
        self.wandb.log({self._object_name(hist_name): hist}, step=self.step)

    def add_table(self, table_name, table):
        self.wandb.log(
            {self._object_name(table_name): self.wandb.Table(dataframe=table)},
            step=self.step,
        )
