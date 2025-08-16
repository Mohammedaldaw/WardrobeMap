import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageOps

import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
import joblib

st.set_page_config(page_title="Fashion-MNIST Unsupervised", layout="wide")

# ---------- small helpers ----------
def to_img(arr01: np.ndarray) -> Image.Image:
    arr = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def preprocess_uploaded(file):
    img = Image.open(file).convert("L")
    img = ImageOps.fit(img, (28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr

@st.cache_resource(show_spinner=False)
def load_artifacts():
    art = Path("artifacts")
    if not art.exists():
        st.error("artifacts/ not found. Run the artifact cell in the notebook first.")
        st.stop()
    pca = joblib.load(art / "pca_100_whiten.joblib")
    # pick any kmeans_*.joblib
    km_files = list(art.glob("kmeans_*.joblib"))
    if not km_files:
        st.error("kmeans_*.joblib not found in artifacts/.")
        st.stop()
    km = joblib.load(km_files[0])
    td = np.load(art / "train_data.npz")
    sc = np.load(art / "app_scatter.npz")
    labels = json.load(open(art / "labels.json"))
    c2y = joblib.load(art / "cluster_majority_map.joblib")
    return pca, km, td, sc, labels, c2y

@st.cache_data(show_spinner=False)
def load_train_images():
    # try Keras first; fallback to OpenML
    try:
        from tensorflow.keras.datasets import fashion_mnist
        (Xtr, ytr), _ = fashion_mnist.load_data()
        Xtr = Xtr.astype("float32") / 255.0
        return Xtr, ytr
    except Exception:
        ds = fetch_openml(data_id=40996, as_frame=False)
        X = ds["data"].astype("float32") / 255.0
        y = ds["target"].astype(int)
        return X[:60000].reshape(-1, 28, 28), y[:60000]

def heatmap_grid(imgs, titles, main_title="", rows=2, cols=5):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    r = c = 1
    for img in imgs:
        vmax = 255 if img.max() > 1 else 1
        fig.add_trace(
            go.Heatmap(z=img, colorscale="gray", zmin=0, zmax=vmax, showscale=False),
            row=r, col=c
        )
        c += 1
        if c == cols + 1:
            r, c = r + 1, 1
    fig.update_layout(height=520, title_text=main_title, margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
    return fig

# ---------- load everything ----------
pca, km, td, sc, LABELS, c2y = load_artifacts()
Xtr, ytr = load_train_images()
Z_train = td["Z_train"]
train_clusters = td["train_clusters"]
y_train = td["y_train"]

# neighbors on PCA space
@st.cache_resource(show_spinner=False)
def get_nn():
    return NearestNeighbors(n_neighbors=50, metric="euclidean").fit(Z_train)
nn = get_nn()

# ---------- TABS ----------
tab_report, tab_viz, tab_demo, tab_readme = st.tabs(
    ["Report", "Visualizations", "Demo", "README"]
)

# ==================== REPORT (submit as PDF) ====================
with tab_report:
    st.title("Fashion-MNIST: Unsupervised Clustering")

    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Goal")
        st.markdown(
            "- Uncover structure without labels using **PCA/SVD** + **clustering**.\n"
            "- Use labels only to **interpret** clusters (heatmaps, purity)."
        )
        st.subheader("Data")
        st.markdown("28×28 grayscale, train **60k**, test **10k** (Fashion-MNIST).")

        # EVR figure
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        k95 = int(np.argmax(cum >= 0.95)) + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=evr, mode="lines", name="EVR"))
        fig.add_trace(go.Scatter(y=cum, mode="lines", name="cumulative", yaxis="y2"))
        fig.update_layout(
            title=f"PCA explained variance (k95≈{k95})",
            xaxis_title="component", yaxis_title="EVR",
            yaxis2=dict(title="cumulative", overlaying="y", side="right", range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Working point")
        st.markdown(
            "- **100 PCs (whitened)** for clustering speed/quality.\n"
            "- K-Means **K ≈ 8–12**; we use the model you exported."
        )
        # training confusion-style heatmap
        ct = pd.crosstab(pd.Series(y_train, name="true"),
                         pd.Series(train_clusters, name="cluster"))
        ct.index = [LABELS[str(i)] if str(i) in LABELS else LABELS[int(i)] for i in ct.index]
        st.markdown("**Cluster mix (train)**")
        st.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto"), use_container_width=True)

    st.subheader("Takeaways")
    st.markdown(
        "- Pixel intensity is skewed to 0 → edges/shapes carry variance.\n"
        "- ~95% variance around ~189 PCs; **100 PCs** works well in practice.\n"
        "- K-Means: strong groups for **Trouser** & **Ankle boot**; **T-shirt vs Shirt** overlap.\n"
        "- DBSCAN is sensitive; not our default here."
    )
    st.info("Submit this tab as PDF: browser **Print → Save as PDF** (landscape works best).")

# ==================== VISUALIZATIONS ====================
with tab_viz:
    st.header("Exploratory visuals")

    # PCA 2D scatter (pre-sampled)
    st.subheader("PCA 2D scatter (sample)")
    df_sc = pd.DataFrame({"pc1": sc["Z2"][:, 0], "pc2": sc["Z2"][:, 1], "cluster": sc["clusters"].astype(int)})
    st.plotly_chart(px.scatter(df_sc, x="pc1", y="pc2", color="cluster", opacity=0.6), use_container_width=True)

    # reconstructions
    st.subheader("Reconstructions")
    one_each = [np.where(ytr == c)[0][0] for c in range(10)]
    orig = Xtr[one_each]
    orig_flat = orig.reshape(len(orig), -1)

    k_list = st.multiselect("k components", [10, 50, 100, 200], default=[10, 50, 100, 200])
    if k_list:
        st.plotly_chart(heatmap_grid(orig, [LABELS.get(str(c), str(c)) for c in range(10)],
                         "Original (one per class)"), use_container_width=True)
        for k in k_list:
            p = joblib.clone(pca) if hasattr(joblib, "clone") else None
            # quick fresh PCA with k, still whiten=False for reconstruction clarity
            from sklearn.decomposition import PCA as SKPCA
            p = SKPCA(n_components=k, svd_solver="randomized", random_state=42).fit(Xtr.reshape(len(Xtr), -1))
            rec = p.inverse_transform(p.transform(orig_flat)).reshape(-1, 28, 28)
            st.plotly_chart(heatmap_grid(rec, [f"{LABELS.get(str(c),str(c))} (k={k})" for c in range(10)],
                             f"PCA reconstructions @ k={k}"), use_container_width=True)

    # cluster purity (train)
    st.subheader("Cluster purity (train)")
    ct = pd.crosstab(pd.Series(y_train, name="true"), pd.Series(train_clusters, name="cluster"))
    pur = (ct.max(axis=0) / ct.sum(axis=0)).rename("purity").reset_index()
    st.plotly_chart(px.bar(pur, x="cluster", y="purity"), use_container_width=True)

# ==================== DEMO ====================
with tab_demo:
    st.header("Image → cluster + similar items")

    left, right = st.columns([1, 1])
    with left:
        mode = st.radio("Input", ["Upload image", "Pick sample"], index=0)
        if mode == "Upload image":
            f = st.file_uploader("Upload grayscale clothing image", type=["png","jpg","jpeg","bmp","webp"])
            if f is None:
                st.stop()
            arr = preprocess_uploaded(f)
            title = "Uploaded image"
        else:
            names = [LABELS.get(str(i), str(i)) for i in range(10)]
            name = st.selectbox("Class", names, index=1)
            cid = [i for i, v in enumerate(names) if v == name][0]
            idx = st.slider("Index within class", 0, int((ytr == cid).sum() - 1), 0)
            row = np.where(ytr == cid)[0][idx]
            arr = Xtr[row]
            title = f"{name} · id {row}"

        k_neighbors = st.slider("Similar items (k)", 4, 24, 12, step=2)

    with right:
        st.markdown(f"**{title}**")
        st.image(to_img(arr), width=180)

        xflat = arr.reshape(1, -1)
        z = pca.transform(xflat)
        c_pred = int(km.predict(z)[0])
        pretty_label = LABELS.get(str(c2y.get(c_pred, -1)), LABELS.get(c2y.get(c_pred, -1), str(c2y.get(c_pred, -1))))

        st.write(f"**Predicted cluster:** {c_pred}  ·  **Majority label:** {pretty_label}")

        # neighbors in PCA space; prefer same cluster if possible
        d, idxs = nn.kneighbors(z, n_neighbors=max(k_neighbors, 20), return_distance=True)
        idxs = idxs[0]
        same = np.where(train_clusters[idxs] == c_pred)[0]
        if len(same) >= k_neighbors:
            idxs = idxs[same[:k_neighbors]]
        else:
            idxs = idxs[:k_neighbors]

        cols = st.columns(min(6, k_neighbors))
        for i, ix in enumerate(idxs):
            cols[i % len(cols)].image(to_img(Xtr[ix]), caption=f"id {ix} · c{int(train_clusters[ix])}", width=120)

    # show query on PCA scatter
    st.subheader("Query on PCA(2)")
    df_sc = pd.DataFrame({"pc1": sc["Z2"][:, 0], "pc2": sc["Z2"][:, 1], "cluster": sc["clusters"].astype(int)})
    fig = px.scatter(df_sc, x="pc1", y="pc2", color="cluster", opacity=0.35)
    # overlay query
    fig.add_trace(go.Scatter(x=[z[0, 0]], y=[z[0, 1]], mode="markers", marker_size=12,
                             marker_symbol="x", marker_color="black", name="query"))
    st.plotly_chart(fig, use_container_width=True)

