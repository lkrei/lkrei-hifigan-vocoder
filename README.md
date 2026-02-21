# HiFi-GAN Vocoder (русский, RUSLAN)

Реализация вокодера HiFi-GAN для синтеза речи на русском языке. Обучен на датасете [RUSLAN](https://ruslan-corpus.github.io/).

Ноутбуки:
- [Demo (Colab)](https://colab.research.google.com/drive/1hfjo5XbylGz5n_f6cv0RPMExKhcBNz4h?usp=sharing) — инференс, прослушивание, скорость
- [Experiments & Report (Colab)](https://colab.research.google.com/drive/13soijKUCS1vU8JcAkfDYbQWVTXIBWWyo?usp=sharing) — анализ качества, визуализации, full TTS pipeline

Логирование: [W&B](https://wandb.ai/aredaw882-hse-university/hifigan_vocoder)

Report: [Report.md](https://github.com/lkrei/lkrei-hifigan-vocoder/blob/main/report.md)

## Установка

```bash
git clone https://github.com/lkrei/lkrei-hifigan-vocoder.git
cd lkrei-hifigan-vocoder
pip install -r requirements.txt
```

## Скачивание чекпоинта

```bash
pip install gdown
mkdir -p results
gdown "https://drive.google.com/uc?id=1vu9cWf_3ewtkNtC_oi9mJaOt6ms8jgFX" -O results/model_best.pth
```

## Инференс (resynthesis)

Подготовить папку с wav-файлами в формате `dir/audio/*.wav`:

```bash
python synthesize.py \
  custom_dir_path=wav_test \
  synthesizer.output_dir=output
```

По умолчанию берётся чекпоинт `results/model_best.pth`. Можно переопределить:

```bash
python synthesize.py \
  custom_dir_path=wav_test \
  synthesizer.checkpoint_path=results/model_best.pth \
  synthesizer.output_dir=output
```

Результат: `output/test/*.wav` — сгенерированные файлы с теми же именами, что и исходные.

## Обучение

```bash
python train.py data_dir=/path/to/RUSLAN/audio
```

Основные параметры (переопределяются через CLI):

```bash
python train.py data_dir=/path/to/audio dataloader.batch_size=8 trainer.epoch_len=1000
python train.py data_dir=/path/to/audio trainer.from_pretrained=results/model_best.pth
python train.py data_dir=/path/to/audio trainer.resume_from=checkpoint-epoch5.pth
```

## Архитектура

Реализация по статье [Kong et al., 2020] Вокодер принимает лог-мел-спектрограмму и генерирует waveform.

### Генератор

Полностью свёрточная сеть. Мел-спектрограмма (80 каналов) проходит через `conv_pre` (7×1), затем 4 слоя апсемплинга через `ConvTranspose1d` с ядрами [16, 16, 4, 4] (stride = kernel // 2). Суммарный апсемплинг: ×8 × ×8 × ×2 × ×2 = ×256, что совпадает с `hop_length`. После каждого апсемплинга — модуль MRF (Multi-Receptive Field Fusion): три параллельных ResBlock с ядрами [3, 7, 11] и dilations [[1,1], [3,1], [5,1]], выходы суммируются и усредняются. Финал: LeakyReLU → `conv_post` (7×1) → tanh.

Конфигурация V1 (`hu=512`), ~13.9M параметров.

### Дискриминаторы

**MPD (Multi-Period Discriminator):** 5 под-дискриминаторов с периодами [2, 3, 5, 7, 11] — простые числа, чтобы минимизировать перекрытие. Входной waveform (1D) перестраивается в 2D тензор (T/p × p), затем обрабатывается 2D свёртками с ядром (k, 1), чтобы каждый канал периода обрабатывался независимо. Weight normalization.

**MSD (Multi-Scale Discriminator):** 3 под-дискриминатора на разных масштабах — raw audio, ×2 average-pooled, ×4 average-pooled. 1D свёртки с группами. Первый под-дискриминатор использует spectral normalization, остальные — weight normalization.

MPD ловит периодические паттерны (что критично для речи — синусоидальные составляющие разных частот), MSD — последовательные зависимости и общую структуру.

### Функции потерь

- **LS-GAN (adversarial):** D обучается классифицировать реальные сэмплы к 1, сгенерированные к 0; G обучается обманывать D. Least squares вместо BCE — стабильнее градиенты.
- **Feature Matching Loss (λ=2):** L1 между промежуточными фичами D для реального и сгенерированного аудио. Стабилизирует обучение G.
- **Mel-Spectrogram Loss (λ=45):** L1 между мел-спектрограммами реального и сгенерированного waveform. Основной reconstruction loss, фокусирует G на перцептуально значимых частотах.

Итого: `L_G = L_adv + 2 * L_fm + 45 * L_mel`, `L_D = L_adv`.

### Mel-спектрограмма

| Параметр | Значение |
|---|---|
| sr | 22050 |
| n_fft | 1024 |
| hop_length | 256 |
| win_length | 1024 |
| n_mels | 80 |
| f_min | 0 |
| f_max | 11025 |

Mel-базис — Slaney (через librosa), логарифмирование: `log(clamp(mel, min=1e-5))`. Один и тот же код используется при обучении и при инференсе.

### Обучение

- Данные: RUSLAN (22200 файлов, мужской голос), ресемплинг 44100→22050 Hz, сегменты по 8192 сэмпла при обучении
- Split: train 90%, val 5%, test 5% (seed=42)
- Оптимизатор: AdamW, lr=2e-4, beta=(0.8, 0.99), weight_decay=0.01
- LR scheduler: ExponentialLR, gamma=0.999
- Batch size: 8
- ~17 эпох, отбор лучшей модели по val_loss_mel, несколько запусков

## Структура

```
├── train.py
├── synthesize.py
├── requirements.txt
├── demo.ipynb
├── experiments_report.ipynb
├── results/                # model_best.pth (скачивается через gdown)
├── wav_test/audio/     # тестовые 1.wav, 2.wav, 3.wav
└── src/
    ├── model/            # Generator, Discriminator (MPD, MSD)
    ├── datasets/           # RuslanDataset, CustomDirDataset, collate
    ├── loss/         # LS-GAN, feature matching, mel L1
    ├── transforms/    # MelSpectrogram (Slaney)
    ├── trainer/                # BaseTrainer, Trainer (D/G steps)
    ├── logger/         # W&B
    ├── metrics/        # MetricTracker
    ├── utils/          # IO, seed, saving
    └── configs/           # Hydra
```
