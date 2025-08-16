# make_artifacts.py
from pathlib import Path
import numpy as np, joblib, json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

def load_fashion():
    try:
        from tensorflow.keras.datasets import fashion_mnist
        (Xtr, ytr), _ = fashion_mnist.load_data()
        Xtr = Xtr.astype("float32")/255.0
        return Xtr, ytr
    except Exception:
        ds = fetch_openml(data_id=40996, as_frame=False)
        X = ds["data"].astype("float32")/255.0
        y = ds["target"].astype(int)
        return X[:60000].reshape(-1,28,28), y[:60000]

LABELS = {
    0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",
    5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"
}

Xtr, ytr = load_fashion()
Xflat = Xtr.reshape(len(Xtr), -1)

print("Fitting PCA(100, whiten=True)…")
pca = PCA(n_components=100, whiten=True, svd_solver="randomized", random_state=42).fit(Xflat)
Z = pca.transform(Xflat)

print("Fitting KMeans(K=10)…")
km = KMeans(n_clusters=10, n_init=20, random_state=42).fit(Z)
clusters = km.predict(Z)

# majority map
import pandas as pd
df = pd.DataFrame({"c":clusters, "y":ytr})
c2y = df.groupby("c")["y"].agg(lambda s: s.value_counts().idxmax()).to_dict()

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(pca, "artifacts/pca_100_whiten.joblib")
joblib.dump(km,  "artifacts/kmeans_10.joblib")
np.savez_compressed("artifacts/train_data.npz", Z_train=Z.astype("float32"),
                    y_train=ytr.astype("int16"), train_clusters=clusters.astype("int16"))
rng = np.random.default_rng(42)
ix = rng.choice(len(Z), size=min(15000, len(Z)), replace=False)
np.savez_compressed("artifacts/app_scatter.npz", Z2=Z[ix,:2].astype("float32"),
                    clusters=clusters[ix].astype("int16"))
json.dump(LABELS, open("artifacts/labels.json","w"))
joblib.dump(c2y, "artifacts/cluster_majority_map.joblib")
print("Saved artifacts/ ✔")
