# GMM-LUAD

Gaussian Mixture Models for Lung Adenocarcinoma Subtype Classification using TCGA expression data.

## 📂 Overview

This project applies three Gaussian Mixture Model (GMM) variants to classify five lung adenocarcinoma (LUAD) subtypes based on gene expression features.

## 🔬 Methods

- **GMM1**: Unsupervised EM-based GMM
- **GMM2**: Supervised GMM (1 Gaussian per class via MLE)
- **GMM3**: Supervised GMM (2 Gaussians per class using EM)

Dimensionality reduction was performed with PCA, and model performance was evaluated using the weighted F1 score across varying numbers of principal components (PCs).

## 📈 Results Summary

- **GMM2** achieved the best performance (F1 = 0.84) with 10 PCs.
- **GMM1** underperformed due to lack of label supervision.
- **GMM3** showed more variance due to overfitting with limited sample size.

## 🗂️ Data

Expression data and subtype labels were derived from:

> Ellrott, K., Wong, C. K., Yau, C., Castro, M. A. A., Lee, J. A., Karlberg, B. J., Grewal, J. K., Lagani, V., Tercan, B., Friedl, V., et al. (2025). *Classification of non-TCGA cancer samples to TCGA molecular subtypes using compact feature sets*. Cancer Cell. https://doi.org/10.1016/j.ccell.2024.12.002
