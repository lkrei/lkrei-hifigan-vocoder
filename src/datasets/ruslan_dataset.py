import logging
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RuslanDataset(Dataset):
    def __init__(
        self,
        audio_dir,
        metadata_path=None,
        target_sr=22050,
        segment_size=8192,
        split="train",
        train_ratio=0.9,
        val_ratio=0.05,
        seed=42,
        limit=None,
        **kwargs,
    ):
        super().__init__()
        self.target_sr = target_sr
        self.segment_size = segment_size
        self.split = split

        audio_dir = Path(audio_dir)
        audio_files = sorted(audio_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found in {audio_dir}")

        rng = random.Random(seed)
        indices = list(range(len(audio_files)))
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train : n_train + n_val]
        else:
            selected = indices[n_train + n_val :]

        self.audio_paths = [audio_files[i] for i in selected]

        if limit is not None:
            self.audio_paths = self.audio_paths[:limit]

        logger.info(
            f"RuslanDataset [{split}]: {len(self.audio_paths)} files "
            f"(segment_size={segment_size}, sr={target_sr})"
        )

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio, sr = torchaudio.load(str(audio_path), backend="soundfile")

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.squeeze(0)

        if self.split == "train":
            if audio.shape[0] >= self.segment_size:
                start = random.randint(0, audio.shape[0] - self.segment_size)
                audio = audio[start : start + self.segment_size]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.shape[0])
                )

        return {
            "audio": audio.unsqueeze(0),
            "audio_path": str(audio_path),
        }
