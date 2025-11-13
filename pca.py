import numpy as np
from dataVisual import print_image

def standarize(image,mean,std,print=False):
    image = np.asarray(image, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = np.where(std == 0, 1.0, std)
    image = ((image - mean) / std_safe).astype(np.float32)
    if print:
        print_image(image)
    return image

def standarize_bulk(images,mean,std):
    result = []
    for image in images:
        result.append(standarize(image,mean,std))
    return np.asarray(result, dtype=np.float32)

def pca(images,og_dataset,mean,std):
    # transformar entradas a matrices 2D (n_muestras, n_pixeles)
    X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    original = np.asarray(og_dataset, dtype=np.float32).reshape(len(og_dataset), -1)
    n_samples = X.shape[0]

    # descomposición en valores singulares sobre datos estandarizados
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    explained_variance = (S**2) / max(n_samples - 1, 1)
    explained_ratio = explained_variance / np.sum(explained_variance)
    cumulative_ratio = np.cumsum(explained_ratio)

    # elegir k por varianza acumulada (95% por defecto)
    k = int(np.searchsorted(cumulative_ratio, 0.95) + 1)
    k = min(k, Vt.shape[0])
    Wk = Vt[:k].T

    # proyección y reconstrucción en el espacio estandarizado
    Z = X @ Wk
    X_rec = Z @ Wk.T

    # volver a la escala original por pixel
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    mean_flat = mean.reshape(-1)
    std_flat = std.reshape(-1)
    std_flat = np.where(std_flat == 0, 1.0, std_flat)
    X_rec_orig = (X_rec * std_flat) + mean_flat

    mse_train = np.mean((original - X_rec_orig)**2)
    print(f"k={k} | MSE train={mse_train:.6f}")

    return X_rec_orig.reshape((-1,) + mean.shape)
