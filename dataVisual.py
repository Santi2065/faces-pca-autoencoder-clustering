import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

"""Lee el CSV y devuelve un array numpy de imágenes en escala de grises
    con forma (N, side, side), tipo float32 y valores en [0,1].
"""
def load_images(path, img_side=64):

    df = pd.read_csv(path)
    if 'person_id' in df.columns:
        df = df.drop('person_id', axis=1)
    arr = df.to_numpy(dtype=np.float32)
    # normalizar si los valores están en 0-255
    if arr.max() > 1.0:
        arr = arr / 255.0
    n_pix = arr.shape[1]
    side = img_side
    if side * side != n_pix:
        side = int(np.sqrt(n_pix))
    imgs = arr.reshape(-1, side, side)
    return imgs

"Funcion para impimir fotos aleatorias de caras.csv"
"Input: amount = numero de imagenes a imprimir"
def src_img_print(path, amount,rand=True,start_id=0):

    # usar la nueva función para obtener las imágenes
    imgs = load_images(path)
    n = len(imgs)
    if rand:
        rows = random.sample(range(0, n), amount)
    else:
        rows = [i for i in range(start_id*10, start_id*10 + amount)]

    # calcular dimensiones del grid (casi cuadrado)
    ncols = int(np.ceil(np.sqrt(amount)))
    nrows = int(np.ceil(amount / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    # normalizar axes a lista
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    for i, rowNum in enumerate(rows):
        img = imgs[rowNum]

        ax = axes_list[i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Cara {rowNum}")

    # apagar ejes sobrantes
    for ax in axes_list[amount:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

"Imprime una imagen en escala de grises"
def print_image(image):
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.show()
    