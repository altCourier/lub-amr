# LUB-AMR

A machine-learning repository for **Automatic Modulation Recognition (AMR)** using simulated 5G NR I/Q signal datasets.

This repository contains the dataset preparation, model training, evaluation, and experimental analysis developed during a summer internship at **Universität zu Lübeck**.

## Overview

The project investigates the recognition of digital modulation schemes from received I/Q samples under different wireless channel conditions.

The workflow is:

**PHYForge-generated data -> dataset preparation -> preprocessing -> model training -> evaluation -> experimental analysis**

The project uses I/Q datasets generated with [PHYForge](https://github.com/altCourier/PHYForge), including 3GPP UMi, UMa, and RMa channel scenarios.

## Supported Modulations

The AMR experiments use the following modulation classes:

* BPSK
* QPSK
* 16-QAM
* 64-QAM
* 256-QAM

1024-QAM was investigated during dataset generation but excluded from the final experiments due to instability observed in the generated data.

## Models

Two neural-network architectures are implemented:

* **BaselineCNN** a convolutional neural-network baseline for modulation classification.
* **CNNLSTM** a CNN-LSTM architecture using the same convolutional backbone while retaining sequential information through a bidirectional LSTM.

Both models operate on real-valued I/Q representations of the received signals.

## Dataset Pipeline

The dataset pipeline provides:

* HDF5 dataset loading
* I/Q conversion and preprocessing
* Per-sample and global normalization
* Domain-specific filtering for UMi, UMa, and RMa
* Stratified train/validation/test splitting
* Dataset integrity checks
* PyTorch `DataLoader` integration

## Evaluation

The framework provides common evaluation tools for:

* Overall classification accuracy
* Accuracy versus SNR
* Confusion matrices
* High-SNR confusion analysis
* Training and validation curves
* Comparison between channel domains and experimental configurations

## Experiments

The experiments investigate how properties of the generated signals affect AMR performance, with particular attention to:

* Constellation normalization
* Channel-gain normalization
* Bandwidth
* Multipath fading
* Inter-symbol interference
* Higher-order QAM classification

The results showed that the physical characteristics of the generated I/Q data can have a substantial effect on AMR performance, particularly for 16-QAM, 64-QAM, and 256-QAM.

## Repository Structure

```text
lub-amr/
├── data/          # Generated and processed datasets
├── checkpoints/   # Saved model checkpoints
├── notebooks/     # Analysis and experiment notebooks
├── reports/       # Experiment reports and results
├── src/           # Dataset, model, training, and evaluation code
└── tests/         # Automated tests
```

## Testing

The project includes automated tests for dataset loading, preprocessing, normalization, labels, and data-loading functionality.

Run the test suite with:

```bash
pytest
```

## Related Project

The PHY-layer simulation and AMR dataset-generation pipeline is maintained separately in:

**[PHYForge](https://github.com/altCourier/PHYForge)**

PHYForge is responsible for generating the simulated communication data, while this repository focuses on preparing that data and using it for AMR machine-learning experiments.

## Internship Context

This repository documents software, experiments, and findings produced during a summer internship at **Universität zu Lübeck**.

It is primarily intended as a record of the development process and experimental work rather than as a general-purpose open-source AMR library.
