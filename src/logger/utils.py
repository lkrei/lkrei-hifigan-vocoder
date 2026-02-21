import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")


def plot_spectrogram(spectrogram, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel channel")

    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(PIL.Image.open(buf))
    plt.close()
    return image
