import logging
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CustomDirDataset(Dataset):
    def __init__(self, dir_path, target_sr=22050, **kwargs):
        super().__init__()
        self.dir_path = Path(dir_path)
        self.target_sr = target_sr

        audio_dir = self.dir_path / "audio"
        transcription_dir = self.dir_path / "transcriptions"

        self.items = []

        if audio_dir.exists():
            audio_files = sorted(audio_dir.glob("*.wav"))
            for af in audio_files:
                stem = af.stem
                txt_path = transcription_dir / f"{stem}.txt"
                transcription = None
                if txt_path.exists():
                    transcription = txt_path.read_text(encoding="utf-8").strip()
                self.items.append(
                    {
                        "audio_path": str(af),
                        "transcription": transcription,
                        "name": stem,
                    }
                )
        elif transcription_dir.exists():
            txt_files = sorted(transcription_dir.glob("*.txt"))
            for tf in txt_files:
                transcription = tf.read_text(encoding="utf-8").strip()
                self.items.append(
                    {
                        "audio_path": None,
                        "transcription": transcription,
                        "name": tf.stem,
                    }
                )

        if not self.items:
            raise FileNotFoundError(
                f"No data found in {self.dir_path}. "
                "Expected audio/ and/or transcriptions/ subdirectories."
            )

        logger.info(f"CustomDirDataset: {len(self.items)} items from {self.dir_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        result = {"name": item["name"]}

        if item["audio_path"] is not None:
            audio, sr = torchaudio.load(item["audio_path"], backend="soundfile")
            if sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, sr, self.target_sr)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            result["audio"] = audio
            result["audio_path"] = item["audio_path"]

        if item["transcription"] is not None:
            result["transcription"] = item["transcription"]

        return result
