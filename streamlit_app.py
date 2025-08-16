# streamlit_app.py  —  WardrobeMap (Report / Visualizations / Demo)

import json
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
from sklearn.decomposition import PCA as SKPCA
import joblib

st.set_page_config(page_title="Fashion-MNIST Unsupervised", layout="wide")

# ---------- helpers ----------
def to_img(arr01: np.ndarray) -> Image.Image:
    """[0,1] -> PIL gray image."""
    arr = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def preprocess_uploaded(file):
    """
    Load an uploaded image and convert it to a 28x28, FMNIST-like tensor in [0,1]:
      - grayscale
      - letterbox to 28x28 (no cropping)
      - auto-invert if background is lighter than the object
      - light contrast rescale
    Returns: np.ndarray shape (28,28) float32 in [0,1]
    """
    # 1) load & grayscale
    pil = Image.open(file).convert("L")
    pil = ImageOps.exif_transpose(pil)

    # 2) letterbox to 28x28 (preserve aspect; pad to square)
    #    First, scale the longer side to 28 without cropping
    pil_scaled = ImageOps.contain(pil, (28, 28))
    w, h = pil_scaled.size

    # Estimate background from original image edges to fill padding
    arr_full = np.asarray(pil, dtype=np.float32)
    edges = np.concatenate([
        arr_full[0, :], arr_full[-1, :], arr_full[:, 0], arr_full[:, -1]
    ])
    bg = int(np.median(edges))  # robust fill value

    canvas = Image.new("L", (28, 28), color=bg)
    offx = (28 - w) // 2
    offy = (28 - h) // 2
    canvas.paste(pil_scaled, (offx, offy))

    # 3) auto-invert if border brighter than center (typical for web product photos)
    arr = np.asarray(canvas, dtype=np.float32)
    border = np.r_[arr[0:4, :].ravel(), arr[-4:, :].ravel(), arr[:, 0:4].ravel(), arr[:, -4:].ravel()]
    center = arr[6:22, 6:22].ravel()

    if border.mean() > center.mean():
        arr = 255.0 - arr  # make foreground bright on dark background

    # 4) contrast rescale (gentle)
    lo, hi = np.percentile(arr, [2, 98])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        arr = arr / 255.0

    return arr.astype("float32")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load models + precomputed arrays saved from the notebook."""
    art = Path("artifacts")
    if not art.exists():
        st.error("artifacts/ not found. Run the artifact-saving cell in the notebook first.")
        st.stop()

    pca = joblib.load(art / "pca_100_whiten.joblib")

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
    """
    Get the 60k training images.
    Tries Keras first; falls back to OpenML if TensorFlow isn't installed.
    """
    try:
        from tensorflow.keras.datasets import fashion_mnist
        (Xtr, ytr), _ = fashion_mnist.load_data()
        Xtr = Xtr.astype("float32") / 255.0
        return Xtr, ytr
    except Exception:
        ds = fetch_openml(data_id=40996, as_frame=False)  # Fashion-MNIST
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

# ---------- load artifacts + data ----------
pca, km, td, sc, LABELS, c2y = load_artifacts()
Xtr, ytr = load_train_images()

Z_train = td["Z_train"]
train_clusters = td["train_clusters"]
y_train = td["y_train"]

@st.cache_resource(show_spinner=False)
def get_nn():
    return NearestNeighbors(n_neighbors=50, metric="euclidean").fit(Z_train)

nn = get_nn()

# ---------- TABS ----------
tab_report, tab_viz, tab_demo = st.tabs(["Report", "Visualizations", "Demo"])

# ==================== REPORT ====================
with tab_report:
    st.title("Fashion-MNIST: Unsupervised Clustering")

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Goal")
        st.markdown(
            "- Explore structure without labels using **PCA/SVD** + **clustering**.\n"
            "- Use labels only afterward to interpret clusters."
        )
        st.subheader("Data")
        st.markdown("28×28 grayscale images, train **60k**, test **10k** (Fashion-MNIST).")

        # Explained variance
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
            "- K-Means **K ≈ 8–12** (model loaded from artifacts)."
        )
        # Cluster mix on train
        ct = pd.crosstab(pd.Series(y_train, name="true"),
                         pd.Series(train_clusters, name="cluster"))
        ct.index = [LABELS.get(str(i), str(i)) for i in ct.index]
        st.markdown("**Cluster mix (train)**")
        st.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto"), use_container_width=True)

    st.subheader("Key points")
    st.markdown(
        "- Pixel intensity is skewed to 0 → edges/shapes carry most variance.\n"
        "- ~95% variance around ~189 PCs; **100 PCs** is a good trade-off.\n"
        "- K-Means: clear groups for **Trouser** & **Ankle boot**; **T-shirt vs Shirt** overlap.\n"
        "- DBSCAN is sensitive to `eps`; not the default here."
    )
    st.info("To submit the report: open this tab → **Print** → **Save as PDF** (landscape works best).")

# ==================== VISUALIZATIONS ====================
with tab_viz:
    st.header("Exploratory visuals")

    # PCA 2D scatter (pre-sampled)
    st.subheader("PCA 2D scatter (sample)")
    df_sc = pd.DataFrame({
        "pc1": sc["Z2"][:, 0],
        "pc2": sc["Z2"][:, 1],
        "cluster": sc["clusters"].astype(int)
    })
    st.plotly_chart(px.scatter(df_sc, x="pc1", y="pc2", color="cluster", opacity=0.6),
                    use_container_width=True)

    # Reconstructions
    st.subheader("Reconstructions (PCA)")
    one_each = [np.where(ytr == c)[0][0] for c in range(10)]
    orig = Xtr[one_each]
    orig_flat = orig.reshape(len(orig), -1)

    k_list = st.multiselect("k components", [10, 50, 100, 200], default=[10, 50, 100, 200])
    if k_list:
        st.plotly_chart(
            heatmap_grid(orig, [LABELS.get(str(c), str(c)) for c in range(10)], "Original (one per class)"),
            use_container_width=True
        )
        for k in k_list:
            p = SKPCA(n_components=k, svd_solver="randomized", random_state=42).fit(Xtr.reshape(len(Xtr), -1))
            rec = p.inverse_transform(p.transform(orig_flat)).reshape(-1, 28, 28)
            st.plotly_chart(
                heatmap_grid(rec, [f"{LABELS.get(str(c),str(c))} (k={k})" for c in range(10)],
                             f"PCA reconstructions @ k={k}"),
                use_container_width=True
            )

    # Cluster purity
    st.subheader("Cluster purity (train)")
    ct = pd.crosstab(pd.Series(y_train, name="true"),
                     pd.Series(train_clusters, name="cluster"))
    pur = (ct.max(axis=0) / ct.sum(axis=0)).rename("purity").reset_index()

    pur_sorted = pur.sort_values("purity", ascending=False).reset_index(drop=True)
    fig = px.bar(pur_sorted, x="cluster", y="purity", title="Cluster purity (sorted)")
    fig.update_xaxes(categoryorder="array", categoryarray=pur_sorted["cluster"].tolist())
    fig.update_layout(yaxis=dict(range=[0, 1]))
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

# ==================== DEMO ====================
with tab_demo:
    from io import BytesIO  # local import

    st.header("Image → cluster + similar items")

    left, right = st.columns([1, 1], gap="large")

    # ---------- LEFT: input ----------
    with left:
        st.subheader("Input")
        mode = st.radio("Source", ["Upload image", "Pick sample"], horizontal=True, index=0)

        if mode == "Upload image":
            f = st.file_uploader(
                "Upload a clothing image (color or grayscale is fine)",
                type=["png", "jpg", "jpeg", "bmp", "webp"]
            )
            if f is None:
                st.stop()

            content = f.read()
            pil_orig = Image.open(BytesIO(content)).convert("L")
            pil_orig = ImageOps.exif_transpose(pil_orig)
            arr_orig = np.asarray(pil_orig).astype("float32") / 255.0

            # model-ready (28×28, [0,1], normalized + auto-invert if needed)
            arr_proc = preprocess_uploaded(BytesIO(content))
            title = "Uploaded image"

        else:
            class_names = [LABELS.get(str(i), str(i)) for i in range(10)]
            sel = st.selectbox("Class", class_names, index=1)
            cid = class_names.index(sel)
            idx = st.slider("Index within class", 0, int((ytr == cid).sum() - 1), 0)
            row = np.where(ytr == cid)[0][idx]

            arr_orig = Xtr[row]   # already 28×28, [0,1]
            arr_proc = arr_orig
            title = f"{sel} · id {row}"

        k_neighbors = st.slider("Similar items (k)", 4, 24, 12, step=2)

    # ---------- RIGHT: previews + prediction + neighbors ----------
    with right:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Uploaded image")
            st.image(to_img(arr_orig), width=200)
        with c2:
            st.caption("Processed (28×28)")
            st.image(to_img(arr_proc), width=200)

        # predict cluster on processed image
        xflat = arr_proc.reshape(1, -1)
        z = pca.transform(xflat)
        c_pred = int(km.predict(z)[0])

        # cluster majority (from training-time mapping)
        y_major = c2y.get(c_pred, -1)
        maj_name = LABELS.get(str(y_major), str(y_major))
        st.markdown(f"**Predicted cluster:** {c_pred}  ·  **Cluster majority:** {maj_name}")

        # nearest neighbors in PCA space (prefer same cluster first)
        _, idxs = nn.kneighbors(z, n_neighbors=max(k_neighbors * 3, 30), return_distance=True)
        idxs = idxs[0]
        same_mask = (train_clusters[idxs] == c_pred)
        same = idxs[same_mask]
        rest = idxs[~same_mask]
        picked = np.concatenate([same[:k_neighbors], rest[:max(0, k_neighbors - len(same))]]).astype(int)

        # neighbor vote label (what the nearest training items say)
        nbr_labels = y_train[picked]
        counts = np.bincount(nbr_labels, minlength=10)
        vote_idx = int(np.argmax(counts))
        vote_name = LABELS.get(str(vote_idx), str(vote_idx))
        if counts.sum() > 0:
            st.write(f"**Neighbor vote:** {vote_name} ({counts[vote_idx]}/{counts.sum()})")

        # cluster mix (purity snapshot)
        clus_counts = np.bincount(y_train[train_clusters == c_pred], minlength=10)
        if clus_counts.sum() > 0:
            pct = clus_counts / clus_counts.sum()
            top3 = np.argsort(pct)[::-1][:3]
            mix_str = ", ".join([f"{LABELS.get(str(i), str(i))}: {pct[i]:.0%}" for i in top3 if pct[i] > 0])
            st.caption(f"Cluster {c_pred} mix → {mix_str}")

        # show neighbors
        st.markdown("**Nearest in PCA space**")
        grid = min(6, k_neighbors)
        cols = st.columns(grid)
        for i, ix in enumerate(picked[:k_neighbors]):
            ix = int(ix)
            cols[i % grid].image(
                to_img(Xtr[ix]),
                caption=f"id {ix} · c{int(train_clusters[ix])}",
                width=120
            )

    # ---------- PCA(2) scatter with query ----------
    st.subheader("Query on PCA(2)")
    df_sc = pd.DataFrame({
        "pc1": sc["Z2"][:, 0],
        "pc2": sc["Z2"][:, 1],
        "cluster": sc["clusters"].astype(int),
    })
    fig = px.scatter(df_sc, x="pc1", y="pc2", color="cluster", opacity=0.35)
    fig.add_trace(
        go.Scatter(
            x=[z[0, 0]], y=[z[0, 1]], mode="markers",
            marker_symbol="x", marker_color="black", marker_size=12, name="query"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
