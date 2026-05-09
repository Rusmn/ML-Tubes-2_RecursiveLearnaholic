# ML-Tubes-2_RecursiveLearnaholic

Repository untuk Tugas Besar 2 IF3270 Pembelajaran Mesin.

## Tree Structure

```text
ML-Tubes-2_RecursiveLearnaholic/
├── data/
│   ├── raw/
│   ├── features/
│   └── vocab/
├── docs/
├── models/
│   ├── cnn/
│   └── rnn/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── common/
│   ├── cnn/
│   ├── config/
│   └── rnn/
├── .gitignore
├── README.md
└── requirements.txt
```

## Folder Guide

`data/`
: Folder ini berisi data utama dan artifact yang dipakai ulang. `raw/` menyimpan dataset asli, `features/` menyimpan hasil feature extraction, dan `vocab/` menyimpan vocabulary caption.

Contoh isi:
- `data/raw/flickr8k/`
- `data/features/flickr8k_features.npy`
- `data/vocab/word_to_index.json`

`docs/`
: Folder ini berisi Dokumen proyek.

Contoh isi:
- `docs/laporan.pdf`

`models/`
: Folder ini berisi bobot model hasil training. `cnn/` dipakai untuk model CNN, sedangkan `rnn/` dipakai untuk decoder RNN/LSTM.

Contoh isi:
- `models/cnn/best_cnn.keras`
- `models/rnn/lstm_decoder.npz`

`notebooks/`
: Folder ini berisi notebook eksperimen utama. Digunakan untuk training, evaluasi, visualisasi, dan debugging pipeline.

Contoh isi:
- `01_cnn_training.ipynb`
- `02_caption_preprocessing.ipynb`
- `03_rnn_training.ipynb`

`reports/`
: Folder ini berisi file pendukung laporan. Jika pipeline mengekspor grafik atau tabel, file tersebut dapat disimpan di sini agar lebih rapi.

Contoh isi:
- `reports/figures/cnn_loss.png`
- `reports/tables/rnn_scores.csv`

`src/`
: Folder ini berisi kode reusable yang di-import oleh notebook. `common/` untuk utilitas umum, `cnn/` untuk komponen CNN, `rnn/` untuk komponen recurrent, dan `config/` untuk pengaturan path atau hyperparameter (ini bisa juga via notebook).

Contoh isi:
- `src/common/io.py`
- `src/cnn/layers.py`
- `src/rnn/preprocess.py`

