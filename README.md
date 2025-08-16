# 👗 Fashion-MNIST — Unsupervised Clustering (PCA + K-Means) with Streamlit

End-to-end mini project: reduce image dimensions with **PCA**, cluster with **K-Means**, and ship a **Streamlit** app with tabs for a report, visuals, and a live demo.

---

## 📁 Dataset Overview

**Fashion-MNIST** (Zalando): 70k grayscale images, 28×28 pixels, 10 classes  
(we use 60k for train, 10k for test).

Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

Loaded via **OpenML** (default) or **Keras** (optional).

---

## 🧪 What’s Inside

- **Notebook**: full pipeline (EDA → PCA/SVD → K-Means/DBSCAN → analysis).
- **Artifacts**: saved PCA + K-Means for fast app startup.
- **Streamlit app**: tabs for **Report**, **Visualizations**, **Demo**, **README**.

---

## 🛠️ Tools

- Python • numpy • pandas  
- scikit-learn (PCA, K-Means, DBSCAN, metrics)  
- Plotly (interactive charts)  
- Streamlit (app)  
- pillow, joblib, openml  
- (optional) tensorflow (dataset loader)

---

## 🔬 Phases (Notebook)

1) **Data & Prep**  
   Normalize [0,1], flatten 28×28→784, sanity checks, class prototypes.

2) **Dimensionality Reduction**  
   PCA EVR & cumulative EVR; reconstructions; MSE vs k; SVD compare.

3) **Clustering**  
   PCA(100, whiten) → K-Means (pick K by silhouette). Optional DBSCAN sweep.

4) **Analysis & Interpretation**  
   True×cluster heatmaps, cluster purity, centroid “prototype” images, concise takeaways.

---

## 📊 Sample Insights

- Pixel intensity is skewed to 0 → **edges/shapes** carry most variance.  
- ~95% variance at ~**189 PCs**; **100 PCs (whitened)** is a good working point.  
- K-Means finds clean groups for **Trousers** & **Ankle boots**; **T-shirt vs Shirt** overlaps.  
- DBSCAN is sensitive to `eps`; not the default here.

---

## 📦 Project Structure 
WardrobeMap/
├─ FashionMNIST_Unsupervised_Pipeline.ipynb
├─ streamlit_app.py
├─ artifacts/ # created by the notebook cell (see below)
│ ├─ pca_100_whiten.joblib
│ ├─ kmeans_<K>.joblib
│ ├─ train_data.npz
│ ├─ app_scatter.npz
│ ├─ labels.json
│ └─ cluster_majority_map.joblib
└─ README.md

---

---

## 🚀 Quickstart

### 1) Set up Python env

```bash
pip install --upgrade pip
pip install streamlit plotly scikit-learn pillow joblib openml
# optional dataset loader:
pip install tensorflow
```

#### Windows (PowerShell) venv
```bash
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install streamlit plotly scikit-learn pillow joblib openml tensorflow

```

### 2) Run the Notebook and create artifacts (one time)
Open FashionMNIST_Unsupervised_Pipeline.ipynb and execute all cells.
At the end, run the artifact-saving cell (creates artifacts/ next to the app).


```bash
# from the project folder that contains streamlit_app.py and artifacts/
streamlit run streamlit_app.py
# if Windows launcher is confused:
python -m streamlit run streamlit_app.py

```
Report tab → Print → Save as PDF
Visualizations tab → all plots
Demo tab → interactive app