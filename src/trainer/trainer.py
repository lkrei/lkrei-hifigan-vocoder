import torch

from src.loss import generator_loss, discriminator_loss, feature_matching_loss
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    def __init__(self, mel_spec_config=None, lambda_fm=2, lambda_mel=45, **kwargs):
        super().__init__(**kwargs)

        if mel_spec_config is not None:
            cfg = MelSpectrogramConfig(**mel_spec_config)
        else:
            cfg = MelSpectrogramConfig()
        self.mel_spec = MelSpectrogram(cfg).to(self.device)
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)

        audio_real = batch["audio"]
        mel_real = self.mel_spec(audio_real)

        gen_out = self.generator(mel_real)
        audio_gen = gen_out["audio_gen"]

        min_len = min(audio_real.shape[-1], audio_gen.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_gen = audio_gen[..., :min_len]

        batch["mel_real"] = mel_real
        batch["audio_gen"] = audio_gen

        if self.is_train:
            self.optimizer_d.zero_grad()

            d_real_out, d_real_feat = self.discriminator(audio_real)
            d_fake_out, _ = self.discriminator(audio_gen.detach())

            loss_d = discriminator_loss(d_real_out, d_fake_out)
            loss_d.backward()
            self._clip_grad_norm(self.discriminator)
            self.optimizer_d.step()

            d_real_feat = [
                [f.detach() for f in feats] for feats in d_real_feat
            ]

            self.optimizer_g.zero_grad()

            d_fake_out_g, d_fake_feat_g = self.discriminator(audio_gen)

            loss_adv = generator_loss(d_fake_out_g)
            loss_fm = feature_matching_loss(d_real_feat, d_fake_feat_g)
            mel_gen = self.mel_spec(audio_gen)
            loss_mel = torch.nn.functional.l1_loss(mel_real, mel_gen)

            loss_g = loss_adv + self.lambda_fm * loss_fm + self.lambda_mel * loss_mel
            loss_g.backward()
            self._clip_grad_norm(self.generator)
            self.optimizer_g.step()

            batch["loss_d"] = loss_d
            batch["loss_g"] = loss_g
            batch["loss_adv"] = loss_adv
            batch["loss_fm"] = loss_fm
            batch["loss_mel"] = loss_mel
        else:
            with torch.no_grad():
                mel_gen = self.mel_spec(audio_gen)
                loss_mel = torch.nn.functional.l1_loss(mel_real, mel_gen)
                batch["loss_mel"] = loss_mel
                batch["loss_d"] = torch.tensor(0.0)
                batch["loss_g"] = loss_mel
                batch["loss_adv"] = torch.tensor(0.0)
                batch["loss_fm"] = torch.tensor(0.0)

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        self.writer.add_audio(
            "audio_gen",
            batch["audio_gen"][0],
            sample_rate=self.mel_spec.config.sr,
        )
        self.writer.add_audio(
            "audio_real",
            batch["audio"][0],
            sample_rate=self.mel_spec.config.sr,
        )
