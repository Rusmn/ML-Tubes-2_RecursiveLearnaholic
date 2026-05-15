<h1 align="center">Tubes 2 Machine Learning - CNN and RNN Image Captioning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pip-dependencies-green.svg?logo=pypi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange.svg?logo=tensorflow&logoColor=white"/>
</p>

## Description

This repository contains the implementation for Tugas Besar 2 IF3270 Pembelajaran Mesin. The project focuses on Convolutional Neural Network (CNN), Simple Recurrent Neural Network (Simple RNN), and Long Short-Term Memory (LSTM) models.

The assignment consists of two main tasks. The first task implements CNN forward propagation from scratch and compares it with Keras models on the Intel Image Classification dataset. The second task builds an image captioning pipeline on Flickr8k using a CNN encoder and Simple RNN/LSTM decoders with the pre-inject architecture from Show and Tell.

## Features

- CNN utility functions for image loading, batch loading, and feature extraction  
- CNN forward propagation from scratch with shared and non-shared parameters  
- Required CNN layers: `Conv2D`, `LocallyConnected2D`, pooling layers, global pooling layers, `Flatten`, dense layer, `ReLU`, and `Softmax`  
- Keras CNN training pipeline for 16 architecture variations  
- Macro F1-score evaluation for CNN classification  
- Caption preprocessing for Flickr8k, including cleaning, tokenization, vocabulary building, and padding  
- Simple RNN and LSTM forward propagation from scratch  
- Required recurrent layers: `EmbeddingLayer`, `SimpleRNNCell`, `LSTMCell`, `DenseProjectionLayer`, and `DenseOutputLayer`  
- Keras pre-inject decoder for Simple RNN and LSTM  
- Training grid for 6 Simple RNN variations and 6 LSTM variations  
- Greedy caption decoding, BLEU-4, METEOR, runtime evaluation, and weight export to `.npz`  

The reusable implementation is placed in `src`, while notebooks are used only to run experiments, show visualizations, and save result artifacts.

## Project Structure

- `src/cnn` - CNN utilities, feature extraction, and CNN forward propagation modules  
- `src/rnn` - caption preprocessing, recurrent layers, decoder training, decoding, evaluation, and weight export  
- `src/common` - shared utilities for file I/O, metrics, and automatic differentiation  
- `src/model_implementation` - reusable FFNN components from the previous assignment  
- `notebooks` - numbered experiment notebooks and the combined `main.ipynb` pipeline  
- `data/raw` - raw datasets such as Intel Image Classification and Flickr8k  
- `data/features` - extracted Flickr8k image features  
- `data/vocab` - vocabulary and encoded caption sequences  
- `models/cnn` - trained CNN models and weights  
- `models/rnn` - trained decoder models and exported `.npz` weights  
- `reports/figures` - saved plots and qualitative visualization outputs  
- `reports/tables` - saved scores, histories, and comparison tables  
- `doc` - final report artifacts when required for submission  

## Experiments

The experiment workflow is organized into numbered notebooks:

1. CNN training and evaluation (`01_cnn_training.ipynb`)  
2. Caption preprocessing (`02_caption_preprocessing.ipynb`)  
3. Flickr8k feature extraction (`03_feature_extraction.ipynb`)  
4. Simple RNN and LSTM decoder training (`04_rnn_lstm_training.ipynb`)  
5. Evaluation and analysis (`05_evaluation_and_analysis.ipynb`)  
6. End-to-end combined pipeline (`main.ipynb`)  

The required experiments include:

1. Training 16 Keras CNN architectures with Conv2D shared parameters  
2. Comparing CNN Keras inference with CNN forward propagation from scratch  
3. Comparing shared `Conv2D` and non-shared `LocallyConnected2D` CNN variants  
4. Training 6 Simple RNN decoders and 6 LSTM decoders  
5. Measuring BLEU-4, METEOR, and runtime for each decoder variation  
6. Comparing Keras and from-scratch decoder inference  
7. Comparing Simple RNN and LSTM caption quality  
8. Testing at least 3 maximum caption length variations  
9. Showing qualitative analysis for at least 10 images with generated and ground truth captions  

## Documentation

The assignment specification requires a report containing the problem description, implementation explanation, forward propagation explanation, experiment results, conclusion, task distribution, references, and AI usage form.

The README provides the repository overview, setup instructions, run instructions, and task distribution. The final report should be placed in the documentation folder when prepared for submission.

## Requirements

1. Python 3.10 or higher
2. pip (Python package manager)
3. Virtual environment (very recommended, like venv or conda)
4. Jupyter Notebook or another notebook runner

## Python Dependencies (requirements.txt)

```bash
# Core scientific stack
numpy
pandas
matplotlib
scikit-learn
pillow

# Deep learning and NLP
tensorflow
nltk

# Notebook workflow
jupyter
ipykernel
```

## Getting Started

1. Clone Repository

    ```bash
    git clone https://github.com/Rusmn/ML-Tubes-2_RecursiveLearnaholic.git

    cd ML-Tubes-2_RecursiveLearnaholic
    ```

2. Install dependencies

    This step is only necessary if you plan to run in local environment.

    ```bash
    pip install -r requirements.txt
    ```

3. Prepare datasets and output folders

    Place the required datasets in `data/raw`.

    ```text
    data/raw/intel_image_classification/
    data/raw/flickr8k/
    ```

    The pipeline writes reusable artifacts to `data/features`, `data/vocab`, `models`, and `reports`.

4. Running the program

    Choose the desired notebook to run from `notebooks` and run it via any notebook running environment, like `jupyter notebook`, `deepnote`, `google colab`, etc.

    The recommended order is:

    ```text
    notebooks/01_cnn_training.ipynb
    notebooks/02_caption_preprocessing.ipynb
    notebooks/03_feature_extraction.ipynb
    notebooks/04_rnn_lstm_training.ipynb
    notebooks/05_evaluation_and_analysis.ipynb
    notebooks/main.ipynb
    ```

5. Optional source validation

    ```bash
    python3 -m compileall -q src
    ```

## Creators

### Group - RecursiveLearnaholic

<table>
    <tr align="left">
        <td><b>Member</b></td>
        <td><b>Responsibilities</b></td>
    </tr>
    <tr align="left">
        <td>Member 1</td>
        <td>CNN implementation, CNN utility functions, feature extractor, Keras CNN training, CNN forward propagation from scratch, macro F1-score evaluation, shared vs non-shared CNN analysis, and CNN report section.</td>
    </tr>
    <tr align="left">
        <td>Member 2</td>
        <td>Caption preprocessing, vocabulary preparation, Simple RNN and LSTM layers from scratch, embedding layer, dense projection/output layers, Keras weight loading, and RNN/LSTM implementation explanation.</td>
    </tr>
    <tr align="left">
        <td>Member 3</td>
        <td>Image captioning pipeline, Keras pre-inject decoder, decoder training grid, weight export, greedy decoding, BLEU-4 and METEOR evaluation, runtime measurement, length study, qualitative analysis, and RNN/LSTM experiment report section.</td>
    </tr>
</table>
