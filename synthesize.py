import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    set_random_seed(config.synthesizer.seed)

    if config.synthesizer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.synthesizer.device

    generator = instantiate(config.model).to(device)

    checkpoint_path = config.synthesizer.checkpoint_path
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    elif "state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["state_dict"])
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()
    generator.remove_weight_norm()

    mel_cfg = MelSpectrogramConfig(**OmegaConf.to_container(config.mel_spectrogram))
    mel_spec = MelSpectrogram(mel_cfg).to(device)

    output_dir = Path(config.synthesizer.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataloaders, _ = get_dataloaders(config, device)

    for part_name, dataloader in dataloaders.items():
        part_dir = output_dir / part_name
        part_dir.mkdir(exist_ok=True, parents=True)

        for batch in dataloader:
            names = batch.get("name", [])

            if "audio" in batch:
                audio = batch["audio"].to(device)
                mel = mel_spec(audio)
            else:
                raise NotImplementedError(
                    "TTS mode requires an external acoustic model."
                )

            with torch.no_grad():
                gen_out = generator(mel)
                audio_gen = gen_out["audio_gen"]

            for i, name in enumerate(names):
                wav = audio_gen[i].cpu()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                out_path = part_dir / f"{name}.wav"
                torchaudio.save(str(out_path), wav, mel_cfg.sr)
                print(f"  Saved: {out_path}")

    print(f"\nSynthesis complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
