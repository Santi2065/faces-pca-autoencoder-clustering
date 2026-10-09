"""Regenerates the figures and numbers shown in the README.

Uses the repository's own modules (pca.py, kMeans.py, GMM.py, metrics.py) and the
saved artifacts (modelo_pca.npz, autoencoder_best.pth, clustering_datasets.npz).
The autoencoder is evaluated with a small NumPy re-implementation of its forward
pass, so PyTorch is not needed.

    pip install numpy pandas matplotlib
    python docs/figures/make_figures.py      # about one minute
"""
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]

sys.path.insert(0, str(ROOT))
from dataVisual import load_images  # noqa: E402
from GMM import GMM  # noqa: E402
from kMeans import KMeans  # noqa: E402
from metrics import (adjusted_rand_index, homogeneity_completeness_v_measure,  # noqa: E402
                     normalized_mutual_info_score, silhouette_score)
from pca import standarize_bulk  # noqa: E402


def save(fig, name, **kw):
    fig.savefig(HERE / name, metadata={"Date": None} if name.endswith(".svg") else {}, **kw)
    plt.close(fig)


# ---- data and split, exactly as in the notebook (seeded 80/20 permutation)
images = load_images(ROOT / "data/caras.csv", 64)
import pandas as pd  # noqa: E402

ids = pd.read_csv(ROOT / "data/caras.csv", usecols=["person_id"])["person_id"].to_numpy()
np.random.seed(42)
perm = np.arange(len(images))
np.random.shuffle(perm)
test_idx, train_idx = perm[:80], perm[80:]
train, test, y_train, y_test = images[train_idx], images[test_idx], ids[train_idx], ids[test_idx]
mean, std = train.mean(0), train.std(0)
train_std, test_std = standarize_bulk(train, mean, std), standarize_bulk(test, mean, std)

saved = np.load(ROOT / "modelo_pca.npz")
data = np.load(ROOT / "clustering_datasets.npz")
assert np.allclose(saved["mean"], mean) and np.array_equal(data["y_test"], y_test)

# ---- PCA (SVD of the standardized training images)
X = train_std.reshape(len(train_std), -1)
_, S, Vt = np.linalg.svd(X, full_matrices=False)
ratio = S**2 / np.sum(S**2)
cum = np.cumsum(ratio)
k90, k95 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1)
Wk = saved["Wk"]
assert Wk.shape[1] == k90
sd_safe = np.where(std == 0, 1.0, std)


def from_std(Z):
    return Z.reshape(-1, 64, 64) * sd_safe + mean


Xt = test_std.reshape(len(test_std), -1)
pca_rec_std = (Xt @ Wk) @ Wk.T


# ---- autoencoder: load autoencoder_best.pth without torch and run its forward pass
def load_state_dict(path):
    zf = zipfile.ZipFile(path)
    pre = zf.namelist()[0].split("/")[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return lambda st, off, size, stride, *_: np.lib.stride_tricks.as_strided(
                    st[off:], size, [s * st.itemsize for s in stride]).copy()
            if (mod, name) == ("collections", "OrderedDict"):
                return OrderedDict
            if mod == "torch" and name == "FloatStorage":
                return name
            raise pickle.UnpicklingError(f"unexpected {mod}.{name}")

        def persistent_load(self, pid):
            _, _, key, _, n = pid
            return np.frombuffer(zf.read(f"{pre}/data/{key}"), np.float32)[:n]

    return Unpickler(zf.open(f"{pre}/data.pkl")).load()


W = load_state_dict(ROOT / "autoencoder_best.pth")


def conv(x, w, b):  # nn.Conv2d(kernel 3, stride 2, padding 1)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    h = x.shape[2] // 2
    out = sum(np.einsum("nchw,oc->nohw", xp[:, :, i:i + 2 * h:2, j:j + 2 * h:2], w[:, :, i, j])
              for i in range(3) for j in range(3))
    return out + b[None, :, None, None]


def deconv(x, w, b):  # nn.ConvTranspose2d(kernel 3, stride 2, padding 1, output_padding 1)
    n, _, h, _ = x.shape
    full = np.zeros((n, w.shape[1], 2 * h + 1, 2 * h + 1), np.float32)
    for i in range(3):
        for j in range(3):
            full[:, :, i:i + 2 * h:2, j:j + 2 * h:2] += np.einsum("nchw,co->nohw", x, w[:, :, i, j])
    return full[:, :, 1:1 + 2 * h, 1:1 + 2 * h] + b[None, :, None, None]


def relu(x):
    return np.maximum(x, 0)


def encode(x):
    for i in (0, 2, 4):
        x = relu(conv(x, W[f"encoder_conv.{i}.weight"], W[f"encoder_conv.{i}.bias"]))
    return x.reshape(len(x), -1) @ W["encoder_fc.weight"].T + W["encoder_fc.bias"]


def decode(z):
    x = (z @ W["decoder_fc.weight"].T + W["decoder_fc.bias"]).reshape(-1, 64, 8, 8)
    for i in (0, 2, 4):
        x = deconv(x, W[f"decoder_conv.{i}.weight"], W[f"decoder_conv.{i}.bias"])
        x = relu(x) if i < 4 else x
    return x[:, 0]


z_test = encode(test_std[:, None])
assert np.allclose(z_test, data["X_test_ae"], atol=5e-2), "AE re-implementation does not match the saved latents"
ae_rec_std = decode(z_test).reshape(len(test), -1)

pca_rec, ae_rec = from_std(pca_rec_std), from_std(ae_rec_std)
mse = {
    "PCA": (np.mean((Xt - pca_rec_std) ** 2), np.mean((test - pca_rec) ** 2)),
    "AE": (np.mean((Xt - ae_rec_std) ** 2), np.mean((test - ae_rec) ** 2)),
}

# ---- clustering on the saved latent codes (notebook cell "Clustering Automatizado")
KS = list(range(5, 21))
SPACES = {"PCA": (data["X_train_pca"], data["X_test_pca"]), "AE": (data["X_train_ae"], data["X_test_ae"])}
MODELS = {"k-means": KMeans, "GMM": GMM}
sil, results, test_labels = {}, [], {}
for sp, (Xtr, Xte) in SPACES.items():
    for mn, M in MODELS.items():
        s = []
        for k in KS:
            m = M(n_clusters=k, random_state=42)
            m.fit(Xtr)
            s.append(silhouette_score(Xtr, m.predict(Xtr)))
        sil[mn, sp] = np.array(s)
        k = KS[int(np.argmax(s))]
        m = M(n_clusters=k, random_state=42)
        m.fit(Xtr)
        lab = m.predict(Xte)
        test_labels[mn, sp] = lab
        h, c, _ = homogeneity_completeness_v_measure(y_test, lab)
        results.append((f"{mn} ({sp})", k, adjusted_rand_index(y_test, lab), normalized_mutual_info_score(y_test, lab), h, c))

# ---- Figure 1: explained variance and eigenfaces
fig = plt.figure(figsize=(7.2, 2.7))
gs = fig.add_gridspec(2, 5, width_ratios=[2.2, 0.15, 1, 1, 1], wspace=0.08, hspace=0.25)
ax = fig.add_subplot(gs[:, 0])
n = np.arange(1, len(cum) + 1)
ax.plot(n, cum * 100, color=C[0])
for k, lvl, c, lab in ((k90, 90, C[1], "90% (chosen)"), (k95, 95, C[5], "95% (default in pca.py)")):
    ax.plot([k, k, 0], [0, lvl, lvl], color=c, ls=":", lw=0.9, label=f"{lab}: $k={k}$")
ax.legend(loc="lower right")
ax.set_xlim(0, len(cum))
ax.set_ylim(0, 102)
ax.set_xlabel("number of components $k$")
ax.set_ylabel("cumulative explained variance [%]")
ax.set_title("(a) Explained variance")
tiles = [("mean face", mean, "gray")] + [(f"PC {i + 1}", Vt[i].reshape(64, 64), "RdBu_r") for i in range(5)]
for t, (lab, img, cmap) in enumerate(tiles):
    a = fig.add_subplot(gs[t // 3, 2 + t % 3])
    v = np.abs(img).max()
    a.imshow(img, cmap=cmap, **({"vmin": 0, "vmax": 1} if cmap == "gray" else {"vmin": -v, "vmax": v}))
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(lab, fontsize=8, labelpad=2)
    if t == 1:
        a.set_title("(b) Mean face and eigenfaces")
save(fig, "fig1-pca.svg", dpi=150)

# ---- Figure 2: reconstructions of test faces
pick = np.arange(0, 80, 10)
fig, axes = plt.subplots(3, len(pick), figsize=(7.2, 3.0))
rows = (("original", test), (f"PCA, $k={k90}$", pca_rec), (f"AE, latent {k90}", ae_rec))
for r, (lab, imgs) in enumerate(rows):
    for c, i in enumerate(pick):
        a = axes[r, c]
        a.imshow(np.clip(imgs[i], 0, 1), cmap="gray", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    axes[r, 0].set_ylabel(lab, fontsize=8.5)
fig.subplots_adjust(wspace=0.04, hspace=0.06)
save(fig, "fig2-reconstructions.png", dpi=170)

# ---- Figure 3: model selection by silhouette
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, sp, title in zip(axes, ("PCA", "AE"), ("(a) PCA latent space", "(b) Autoencoder latent space")):
    for (mn, c) in (("k-means", C[0]), ("GMM", C[1])):
        s = sil[mn, sp]
        ax.plot(KS, s, color=c, marker="o", ms=3, label=mn)
        b = int(np.argmax(s))
        ax.plot(KS[b], s[b], "o", ms=8, mfc="none", mec=c, mew=1)
    ax.set_xlabel("number of clusters $K$")
    ax.set_title(title)
    ax.set_xticks(range(5, 21, 3))
axes[0].set_ylabel("silhouette score (train)")
axes[0].legend(loc="lower right")
fig.tight_layout()
save(fig, "fig3-silhouette.svg")

# ---- Figure 4: identity-by-cluster contingency on the test set, best model by ARI
best = max(results, key=lambda r: r[2])
mn, sp = best[0].split(" (")[0], best[0].split("(")[1].rstrip(")")
lab = test_labels[mn, sp]
clusters = sorted(np.unique(lab), key=lambda c: -np.sum(lab == c))
persons = sorted(np.unique(y_test), key=lambda p: (clusters.index(np.bincount(lab[y_test == p]).argmax()), p))
M = np.array([[np.sum((y_test == p) & (lab == c)) for p in persons] for c in clusters])
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.imshow(np.ma.masked_equal(M, 0), cmap="Blues", vmin=-1.5, vmax=M.max(), aspect="auto", interpolation="nearest")
for r, c in np.argwhere(M > 1):
    ax.text(c, r, M[r, c], ha="center", va="center", color="white", fontsize=6.5)
ax.set_xticks(range(len(persons)), [str(p) for p in persons], fontsize=5.5)
ax.set_yticks(range(len(clusters)), [f"{np.sum(lab == c)}" for c in clusters], fontsize=6.5)
ax.tick_params(top=False, right=False, length=0)
ax.set_xlabel("person id (test set, 80 images of %d people)" % len(persons))
ax.set_ylabel("cluster (size)")
save(fig, "fig4-clusters.svg")

# ---- numbers for the README
print(f"images {images.shape}, persons {len(np.unique(ids))}, train {len(train)}, test {len(test)}")
print(f"k(90%)={k90} cum={cum[k90 - 1]:.4f} | k(95%)={k95} | PC1 {ratio[0]:.3f} PC1-5 {cum[4]:.3f}")
print("max |z_test - saved X_test_ae| =", np.abs(z_test - data["X_test_ae"]).max())
for k, (a, b) in mse.items():
    print(f"{k}: test MSE standardized={a:.4f}  pixel [0,1]={b:.5f}")
for (mn, sp), s in sil.items():
    print(f"silhouette {mn:8s} {sp:4s}", " ".join(f"{v:.4f}" for v in s), "best K", KS[int(np.argmax(s))])
print("| Model | K | ARI | NMI | Homogeneity | Completeness |")
for r in results:
    print(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} |")
print("best by ARI:", best[0], "| cluster sizes:", [int(np.sum(lab == c)) for c in clusters])
print("clusters that are a single person:", int(np.sum((M > 0).sum(1) == 1)), "| people split over >1 cluster:",
      int(np.sum((M > 0).sum(0) > 1)), "of", len(persons), "| test images per person:", np.bincount(np.bincount(y_test)[persons]).tolist())
