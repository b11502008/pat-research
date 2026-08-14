import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

def gaussian_radon_R(image, thetas, s_values, lam):
    """
    lam 越大，光束越細。
    """
    H, W = image.shape
    num_thetas = len(thetas)
    num_s = len(s_values)
    
    sinogram = np.zeros((num_thetas, num_s))
    
    # 建立圖片像素的座標網格 (以圖片中心為 0,0)
    x1, x2 = np.meshgrid(np.arange(W) - W/2, np.arange(H) - H/2)
    
    # 常數項
    C = np.sqrt(lam / (2 * np.pi))
    
    for i, theta in enumerate(thetas):
        # 預先算好該角度的法向量
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # 算出圖片上每個點在法向量上的投影長度: <theta, x>
        # projection 是一個跟圖片一樣大的 2D 矩陣
        projection = x1 * cos_t + x2 * sin_t
        
        for j, s in enumerate(s_values):
            #  f(y, x) = <theta, x> - s
            f_yx = projection - s
            
            # exp( -lambda/2 * f(y,x)^2 )
            weight = np.exp(-(lam / 2.0) * (f_yx**2))
            
            # 積分：積分( u(x) * weight * C )
            integral = np.sum(image * weight) * C
            sinogram[i, j] = integral
            
    # 習慣上回傳 (s, theta) 的形狀
    return sinogram.T

def gaussian_adjoint_R_star(sinogram, image_shape, thetas, s_values, lam):
    
    H, W = image_shape
    reconstruction = np.zeros((H, W))
    
    x1, x2 = np.meshgrid(np.arange(W) - W/2, np.arange(H) - H/2)
    C = np.sqrt(lam / (2 * np.pi))
    
    for i, theta in enumerate(thetas):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        projection = x1 * cos_t + x2 * sin_t
        
        for j, s in enumerate(s_values):
            # extract point from sinogram
            v_y = sinogram[j, i]
            
            # calculate its Gaussian approach
            f_yx = projection - s
            weight = np.exp(-(lam / 2.0) * (f_yx**2))
            
            # 噴回畫布上
            reconstruction += v_y * weight * C
            
    # 因為我們是把連加當積分，要乘以 dx (也就是 d_theta * d_s)
    # 這裡省略了嚴謹的面積元素以簡化視覺
    return reconstruction

# ==========================================
# 執行與測試 (Try diff. lambda)
# ==========================================

# 1. 準備一張極小的測試圖 (16x16)
SIZE = 16
image = np.zeros((SIZE, SIZE))
image[5:11, 5:11] = 1.0 # 中間畫個方塊

# 2. 定義 y 空間 (theta 和 s)
thetas = np.linspace(0, np.pi, 30, endpoint=False) # 30 個角度
s_values = np.linspace(-SIZE, SIZE, 40)            # 40 個偏移量

# 3. 測試不同的 Lambda
lam_small = 1  # 小 lambda：手電筒光束很粗，圖會很糊
lam_large = 10.0 # 大 lambda：雷射光很細，逼近嚴格的 Dirac Delta

# 執行 Forward
sino_small = gaussian_radon_R(image, thetas, s_values, lam=lam_small)
sino_large = gaussian_radon_R(image, thetas, s_values, lam=lam_large)

# 執行 Adjoint (Backprojection)
recon_small = gaussian_adjoint_R_star(sino_small, image.shape, thetas, s_values, lam=lam_small)
recon_large = gaussian_adjoint_R_star(sino_large, image.shape, thetas, s_values, lam=lam_large)

# 4. 畫圖比較
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# 上排：小 Lambda
axs[0,0].imshow(image, cmap='gray'); axs[0,0].set_title("Original")
axs[0,1].imshow(sino_small, cmap='gray', aspect='auto'); axs[0,1].set_title(fr"R (Sinogram) $\lambda={lam_small}$ (Fat/Smooth)")
axs[0,2].imshow(recon_small, cmap='gray'); axs[0,2].set_title(fr"R* (Recon) $\lambda={lam_small}$")

# 下排：大 Lambda
axs[1,0].imshow(image, cmap='gray'); axs[1,0].set_title("Original")
axs[1,1].imshow(sino_large, cmap='gray', aspect='auto'); axs[1,1].set_title(fr"R (Sinogram) $\lambda={lam_large}$ (Sharp)")
axs[1,2].imshow(recon_large, cmap='gray'); axs[1,2].set_title(fr"R* (Recon) $\lambda={lam_large}$")

plt.tight_layout()
plt.show()
