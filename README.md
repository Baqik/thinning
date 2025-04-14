# thinning

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, skeletonize, thin
from skimage.util import invert
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage.morphology import square
from google.colab import files

# Fungsi Hit-or-Miss
def hit_or_miss(image, se_foreground, se_background):
    image_complement = invert(image)
    eroded_foreground = erosion(image, se_foreground)
    eroded_background = erosion(image_complement, se_background)
    hitmiss_result = eroded_foreground & eroded_background
    return hitmiss_result

# Upload gambar
uploaded = files.upload()
filename = next(iter(uploaded))
img = io.imread(filename)

# Konversi ke grayscale
if img.ndim == 3:
    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    gray = color.rgb2gray(img)
else:
    gray = img

# Binarisasi dengan threshold Otsu
thresh = threshold_otsu(gray)
binary = gray < thresh

# Resize (opsional)
binary = resize(binary, (128, 128), anti_aliasing=False)

# Structuring Elements untuk Hit-or-Miss
se_foreground = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=bool)

se_background = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=bool)

# Proses Hit-or-Miss
hitmiss_result = hit_or_miss(binary, se_foreground, se_background)

# Proses Thinning
thinned = thin(binary)

# Proses Skeletonization
skeleton = skeletonize(binary)

# Visualisasi Semua
fig, axs = plt.subplots(1, 5, figsize=(18, 4))
axs[0].imshow(binary, cmap='gray')
axs[0].set_title("Binary Image")

axs[1].imshow(hitmiss_result, cmap='gray')
axs[1].set_title("Hit-or-Miss")

axs[2].imshow(invert(binary), cmap='gray')
axs[2].set_title("Complement Image")

axs[3].imshow(thinned, cmap='gray')
axs[3].set_title("Thinning")

axs[4].imshow(skeleton, cmap='gray')
axs[4].set_title("Skeletonization")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
