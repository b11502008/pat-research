import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import radon, iradon

xs = np.linspace(-1, 1, 100, endpoint=True)
X = np.array(np.meshgrid(xs, xs, indexing='ij'))
image = 1.*(((X[0]-0.25)**2 + (X[1]-0.25)**2) < 0.125)

# 3. 設定角度 (0 到 180 度)
theta = np.linspace(0., 180., max(image.shape), endpoint=False)

# 4. 進行 Radon 變換 (現在 image 是矩陣了，不會報錯)
sinogram = radon(image, theta=theta)

# 5. 畫出來看看 (左邊原圖，右邊 Sinogram)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
reversed_radon = iradon(sinogram, theta=theta, filter_name=None)

ax1.set_title("Original (Grayscale)")
ax1.imshow(image, cmap='gray')

ax2.set_title("iradon")
ax2.imshow(reversed_radon, cmap='gray', aspect='auto')

plt.show()
