Alzheimer’s Disease Classification Using ConvNeXt

Course: COMP3710 — Pattern Recognition
Author: Edan Zhao
Difficulty: Hard
Framework: PyTorch
Status: In Development

1. Project Overview

This project investigates the use of ConvNeXt, a modern convolutional neural network architecture, for classifying Alzheimer’s Disease (AD) and Cognitively Normal (CN) subjects using structural brain MRI images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI).

The objective is to develop and evaluate a ConvNeXt-based classification model against a simpler convolutional neural network (CNN) baseline. Beyond classification accuracy, this project investigates the reliability of model confidence, particularly when making incorrect or ambiguous predictions.

2. Research Question

Does ConvNeXt provide more accurate and reliable predictions than a standard CNN baseline, particularly for ambiguous cases where prediction confidence may not reflect actual correctness?

The investigation will consider classification performance, prediction confidence, computational efficiency, and representative failure cases.

3. Dataset

The project uses the ADNI brain MRI dataset provided through the UQ Rangpur computing cluster.

Dataset location:
/home/groups/comp3710/ADNI

The dataset directory contains:

* AD_NC/train/ — Training data
* AD_NC/test/ — Testing data
* meta_data_with_label.json — Dataset metadata and labels

The dataset occupies approximately 574 MB.

The data loading pipeline will be designed to prevent patient-level data leakage between training, validation, and testing sets.

Dataset Integrity and Patient-Level Leakage
An initial audit of the supplied ADNI dataset identified 1,076 unique training scans and 450 unique testing scans. Although no scan identifiers were shared between the provided splits, 216 patient identifiers appeared in both sets. This indicates that the supplied split is not patient-independent and may overestimate generalisation performance. To address this limitation, the project will construct new training, validation, and testing partitions grouped by patient identifier, ensuring that no individual contributes images to multiple partitions.

4. Methodology

Two classification models will be implemented using PyTorch:

4.1 Baseline CNN

A standard convolutional neural network will establish a performance benchmark.

4.2 ConvNeXt

A ConvNeXt-based classifier will be implemented and evaluated against the baseline using the same held-out test data.

5. Evaluation

Both models will be evaluated using:

* Classification accuracy
* Per-class precision and recall
* Macro-averaged F1 score
* Area Under the ROC Curve (AUROC)
* Prediction confidence and calibration
* Parameter count, GPU memory usage, and inference latency

A qualitative analysis of 3–5 representative misclassified or ambiguous cases will also be conducted.

6. Implementation

The project will contain the following files:

* modules.py — Model architectures
* dataset.py — Data loading and preprocessing
* train.py — Model training and evaluation
* predict.py — Inference and visualisation
* README.md — Project documentation

7. Results

Baseline CNN

* 1.17M parameters
* Best validation accuracy: 70.51%
* Best validation loss: 0.6016
* Best epoch: 5
* Training stopped at epoch 10
* Peak allocated VRAM: 1.02 GiB
* Average epoch duration: ~94 seconds
* Evidence of overfitting and unstable validation performance

8. Discussion and Conclusion

To be completed after experimental analysis.

9. Artificial Intelligence Usage Disclosure

To be completed with details of AI tools used, their contributions, and the verification procedures performed.
