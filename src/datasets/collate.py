import torch


def collate_fn(dataset_items: list[dict]):
    result = {}

    has_audio = "audio" in dataset_items[0]
    if has_audio:
        audios = [item["audio"] for item in dataset_items]
        max_len = max(a.shape[-1] for a in audios)
        same_len = all(a.shape[-1] == max_len for a in audios)

        if same_len:
            result["audio"] = torch.stack(audios, dim=0)
        else:
            padded = []
            lengths = []
            for a in audios:
                pad_size = max_len - a.shape[-1]
                padded.append(torch.nn.functional.pad(a, (0, pad_size)))
                lengths.append(a.shape[-1])
            result["audio"] = torch.stack(padded, dim=0)
            result["audio_lengths"] = torch.tensor(lengths)

    if "audio_path" in dataset_items[0]:
        result["audio_path"] = [item["audio_path"] for item in dataset_items]

    if "name" in dataset_items[0]:
        result["name"] = [item["name"] for item in dataset_items]

    if "transcription" in dataset_items[0]:
        result["transcription"] = [
            item.get("transcription") for item in dataset_items
        ]

    return result
