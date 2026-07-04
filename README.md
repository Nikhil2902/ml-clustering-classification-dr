# ML Pipeline — Clustering, Classification & Dimensionality Reduction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> A complete machine learning pipeline covering unsupervised clustering, supervised classification, and dimensionality reduction — applied to two real-world datasets with full analysis and explanation of every design decision.

---

## Contents

- [Overview](#overview)
- [Part 1 — Mall Customers](#part-1--mall-customers-clustering--classification)
- [Part 2 — Breast Cancer](#part-2--breast-cancer-dimensionality-reduction)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Related Projects](#related-projects)

---

## Overview

This project demonstrates a full ML workflow across two problem types:

**Part 1** starts with raw, unlabelled customer data and builds a clustering pipeline from scratch. A custom K-Means algorithm using Manhattan distance segments 200 mall customers into 5 groups. Those cluster labels are then used to train and compare four classification algorithms.

**Part 2** applies two dimensionality reduction techniques — PCA and LDA — to a 30-feature medical dataset, reducing it to 2 and 1 dimensions respectively. The results are visualised and compared to determine which technique produces better class separation for a binary classification problem.

---

## Part 1 — Mall Customers: Clustering & Classification

### The Pipeline

```
Raw CSV  →  Feature Selection  →  StandardScaler  →  Custom K-Means  →  Cluster Labels
                                                                              ↓
                                                            Train/Test Split (70/30)
                                                                              ↓
                                                    Logistic Regression  ─┐
                                                    Decision Tree        ─┤→ Compare metrics
                                                    Random Forest        ─┤
                                                    Naive Bayes          ─┘
```

### Custom K-Means: Manhattan vs Euclidean

Standard K-Means uses Euclidean (L2) distance. This implementation uses **Manhattan (L1) distance** — a deliberate design choice:

| Property | Euclidean (L2) | Manhattan (L1) |
|---|---|---|
| Formula | √Σ(xᵢ − yᵢ)² | Σ\|xᵢ − yᵢ\| |
| Outlier sensitivity | High — squares large differences | Lower — sums absolute differences |
| Centroid update | **Mean** (minimises L2 error) | **Median** (minimises L1 error) |
| Best suited for | Normally distributed, clean data | Skewed distributions, outliers |

**Why median, not mean?**  
When using Manhattan distance, the centroid that minimises the sum of L1 distances to all points in a cluster is the **coordinate-wise median** — not the mean. Using the mean with Manhattan distance is a common mistake that produces suboptimal centroids. This implementation correctly uses the median.

```python
# Correct centroid update for Manhattan distance
new_centroids = np.array([
    np.median(X[self.labels_ == k], axis=0)   # ← median, not mean
    for k in range(self.n_clusters)
])
```

### Classifier Comparison

| Classifier | Key Assumption | Strengths | Limitations |
|---|---|---|---|
| Logistic Regression | Linear decision boundary | Fast, interpretable, calibrated probabilities | Fails on non-linear boundaries |
| Decision Tree | Axis-aligned splits | Non-linear, easy to visualise | Prone to overfitting without pruning |
| Random Forest | Ensemble of 100 trees | Robust, handles noise, feature importance | Less interpretable |
| Naive Bayes | Features are conditionally independent | Very fast, works on small datasets | Assumption rarely holds in practice |

---

## Part 2 — Breast Cancer: Dimensionality Reduction

### Dataset

569 samples · 30 features · Binary target (malignant / benign)  
Features are computed from digitised FNA images of breast masses.  
Source: `sklearn.datasets.load_breast_cancer()`

**Class balance:**
- Malignant (0): 212 samples (37.3%)
- Benign (1): 357 samples (62.7%)

### PCA vs LDA — How They Work

**PCA (Principal Component Analysis)**
1. Standardise features
2. Compute the covariance matrix of the standardised data
3. Find eigenvectors (principal components) ordered by eigenvalue (variance captured)
4. Project data onto the top-k eigenvectors

PCA does **not** use class labels. It finds directions of maximum variance in the data — which may or may not correspond to directions that separate classes.

**LDA (Linear Discriminant Analysis)**
1. Standardise features
2. Compute within-class scatter matrix Sw and between-class scatter matrix Sb
3. Find the projection that maximises Sb / Sw — i.e., maximises class separation
4. For binary classification, produces exactly **1** discriminant (min(n_classes − 1, n_features) = 1)

LDA **uses class labels** during fitting. It is specifically designed to make the two class distributions as different as possible in the projected space.

### Why Standardise Before Both?

Both PCA and LDA are sensitive to feature scale. The breast cancer dataset contains features ranging from 0.001 (`fractal dimension error`) to 2501 (`worst area`). Without standardisation, high-magnitude features dominate the covariance/scatter matrices and produce misleading components.

`StandardScaler` transforms each feature to mean=0 and std=1, giving all features equal weight.

---

## Key Results

### Part 1 — Cluster Quality

| Metric | Value | Interpretation |
|---|---|---|
| Adjusted Rand Index (ARI) | **1.00** | Perfect cluster consistency — labels match true groupings |
| Silhouette Score | **0.51** | Well-defined clusters (above 0.5 threshold) |

### Part 2 — PCA Variance Explained

| Component | Variance Explained | Cumulative |
|---|---|---|
| PC1 | 44.27% | 44.27% |
| PC2 | 18.97% | **63.24%** |

### Part 2 — PCA vs LDA Comparison

| | PCA (2 components) | LDA (1 component) |
|---|---|---|
| Uses class labels | No | **Yes** |
| Class separation | Partial overlap | **Near-perfect** |
| Variance captured | 63.2% of total | 100% of between-class |
| Components needed | 2 | **1** |
| Best for | Exploration, noise reduction | **Classification** |

**Conclusion:** LDA outperforms PCA for this dataset. Because it explicitly uses the class labels to find the most discriminative projection, it achieves near-perfect separation of malignant and benign tumors using a single component — while PCA's two components still leave significant class overlap.

---

## Project Structure

```
ml-clustering-classification-dr/
│
├── README.md
├── ml_clustering_classification_dimensionality_reduction.ipynb
│
├── Mall_Customers.csv                  ← Download from Kaggle (link below)
└── Mall_Customers_Clustered.csv        ← Generated after running Part 1
```

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn scipy matplotlib jupyter
```

### Download the Mall Customers Dataset

```
https://www.kaggle.com/shrutimechlearn/step-by-step-kmeans-explained-in-detail/data
```

Save the file as `Mall_Customers.csv` in the same directory as the notebook.

The Breast Cancer dataset is loaded automatically from scikit-learn — no download needed.

### Run the Notebook

```bash
jupyter notebook ml_clustering_classification_dimensionality_reduction.ipynb
```

Or open in [Google Colab](https://colab.research.google.com/) — no local setup required.

Run cells from top to bottom. Part 1 requires `Mall_Customers.csv`. Part 2 loads data from sklearn automatically.

### Expected Outputs

After running the notebook you will have:
- `Mall_Customers_Clustered.csv` — original data with cluster labels added
- `cluster_visualisation.png` — scatter plots of the 5 customer segments
- `pca_lda_comparison.png` — side-by-side PCA scatter vs LDA histogram
- `pca_scree_plot.png` — variance explained per component

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.x | Language |
| NumPy | 1.x | Custom K-Means implementation, array operations |
| pandas | 2.x | Data loading, manipulation, CSV export |
| scikit-learn | 1.x | StandardScaler, classifiers, PCA, LDA, metrics |
| SciPy | 1.x | `cdist` for Manhattan distance computation |
| Matplotlib | 3.x | All visualisations |
| Jupyter | 7.x | Notebook environment |

---
