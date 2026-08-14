import numpy as np
import matplotlib.pyplot as plt

def sensor_perturb(sx, amplitude=5.0, freq=0.1):
    """
    擾動 1：模擬皮膚表面不平整，跟感測器的水平位置 sx 有關。
    """
    return amplitude * np.sin(freq * sx)

def reflector_perturb(x_coords, y_coords, amplitude=8.0, freq=0.08):
    """
    擾動 2：模擬人體組織聲速不均勻，跟聲波經過的空間座標 (x, y) 有關，用 2D Sin 波來模擬一塊塊的脂肪/肌肉。
    """
    return amplitude * np.sin(freq * x_coords) * np.cos(freq * y_coords)

def epsilon_triangle_jitter(sx, amplitude=3.0, freq=0.5):
    """
    擾動 3：模擬物體表面 (皮膚/感測器接觸面) 的微小三角波起伏 (鋸齒粗糙面)
    利用 arcsin(sin(x)) 來產生尖銳的三角波形。
    
    參數:
    - amplitude: 鋸齒的深度 (擾動大小)
    - freq: 鋸齒的密集度
    """
    # np.arcsin(np.sin(x)) 會產生上下限在 -pi/2 到 pi/2 之間的三角波
    # 我們乘以 (2 / np.pi) 將其標準化為 -1 到 1 之間，再乘上振幅
    triangle_wave = (2 / np.pi) * np.arcsin(np.sin(freq * sx))
    
    return amplitude * triangle_wave

def gaussian_forward_UST(image, sensor_x, r_values, lam, distortion_type=None):
    """
    Gaussian Approximation Forward UST (R)
    lam (lambda): 決定聲波的脈衝寬度 (Pulse width)。lam 越大，波越窄。
    """
    H, W = image.shape
    n_sensors = len(sensor_x)
    num_r = len(r_values)
    data_space = np.zeros((num_r, n_sensors))
    
    # 建立網格 (左上角為 0,0，y軸向下為深度)
    y_coords, x_coords = np.mgrid[:H, :W]
    
    # 高斯常數
    C = np.sqrt(lam / (2 * np.pi))
    
    if distortion_type == 'reflector':
        eps_spatial = reflector_perturb(x_coords, y_coords)

    for i, sx in enumerate(sensor_x):
        # 1. 計算所有像素到這個感測器(sx, 0)的距離
        dists = np.sqrt((x_coords - sx)**2 + y_coords**2)
        
        # 2. 廣播機制：計算所有半徑 r 的 f(y, x)
        # dists 是 (H, W)，變成 (H, W, 1)，None指開一個新維度，但是是空的
        # r_values 是 (num_r,)，變成 (1, 1, num_r)
        # 減完後 f_yx 是一個 (H, W, num_r) 的 3D 矩陣：表對這個感測器而言，每一像素點上，對每個r值的f(y,x)
        f_yx = dists[..., None] - r_values
        f_yx_tilde = f_yx.copy()

        if distortion_type == 'sensor':
            # 感測器擾動：單純加上一個數值 (沿著 r 軸平移)
            eps = sensor_perturb(sx)
            f_yx_tilde += eps
        elif distortion_type == 'reflector':
            f_yx_tilde += eps_spatial[..., None]
        elif distortion_type == 'triangle':
            f_yx_tilde += epsilon_triangle_jitter(sx)
            
        # 3. 計算高斯權重
        # Without Perturbation
        # weights = np.exp(-(lam / 2.0) * (f_yx**2))
        # With Perturbation
        weights = np.exp(-(lam / 2.0) * (f_yx_tilde**2))
        
        # 4. 積分：將圖片乘上權重後，把圖片的 H, W 維度 (axis 0和1) 壓扁加總
        # 剩下的大小剛好是 (num_r,)，塞進數據矩陣裡
        integral = np.sum(image[..., None] * weights, axis=(0, 1)) * C
        data_space[:, i] = integral
        
    return data_space

def gaussian_adjoint_UST(data, image_shape, sensor_x, r_values, lam, distortion_type=None):
    """
    Gaussian Approximation Adjoint UST (R*)
    """
    H, W = image_shape
    reconstruction = np.zeros((H, W))
    n_sensors = len(sensor_x)
    
    y_coords, x_coords = np.mgrid[:H, :W]
    C = np.sqrt(lam / (2 * np.pi))
    
    if distortion_type == 'reflector':
        eps_spatial = reflector_perturb(x_coords, y_coords)

    for i, sx in enumerate(sensor_x):
        # 同樣計算距離與 f(y,x)
        dists = np.sqrt((x_coords - sx)**2 + y_coords**2)
        f_yx = dists[..., None] - r_values
        f_yx_tilde = f_yx.copy()

        if distortion_type == 'sensor':
            f_yx_tilde += sensor_perturb(sx)
        elif distortion_type == 'reflector':
            f_yx_tilde += eps_spatial[..., None]
        elif distortion_type == 'triangle':
            f_yx_tilde += epsilon_triangle_jitter(sx)
        
        # 計算高斯權重 (H, W, num_r)
        # weights = np.exp(-(lam / 2.0) * (f_yx**2))
        weights = np.exp(-(lam / 2.0) * (f_yx_tilde**2))
        
        # 拿取該感測器收到的訊號序列 v(y)，形狀為 (num_r,)
        v_y = data[:, i]
        
        # 反投影積分：把權重乘上對應的訊號強度，然後沿著 r 維度 (axis 2) 加起來
        reconstruction += np.sum(weights * v_y, axis=2) * C
        
    return reconstruction

# ==========================================
# 執行與測試 (比較不同的 Lambda)
# ==========================================

# 1. 建立物體空間
H, W = 100, 100
image = np.zeros((H, W))
image[30, 70] = 1
image[70, 30] = 1

# 2. 定義感測器與半徑空間
n_sensors = 128
sensor_x = np.linspace(0, W, n_sensors)
max_dist = int(np.sqrt(H**2 + W**2))
r_values = np.arange(max_dist) # r 的範圍從 0 到對角線最長距離

# 3. 設定 Lambda 並運算
# lam_small 代表波很寬 (低頻超音波)，lam_large 代表波很窄 (高頻超音波)
lam_small = 1
lam_large = 50.0
lam = 2.0

print("Calculating Forward UST...")
data_small = gaussian_forward_UST(image, sensor_x, r_values, lam=lam_small)
data_large = gaussian_forward_UST(image, sensor_x, r_values, lam=lam_large)

print("Calculating Adjoint UST...")
recon_small = gaussian_adjoint_UST(data_small, image.shape, sensor_x, r_values, lam=lam_small)
recon_large = gaussian_adjoint_UST(data_large, image.shape, sensor_x, r_values, lam=lam_large)

print("Calculating Ideal...")
data_ideal = gaussian_forward_UST(image, sensor_x, r_values, lam, None)
recon_ideal = gaussian_adjoint_UST(data_ideal, image.shape, sensor_x, r_values, lam, None)

print("Calculating Sensor Jitter (Bumpy Skin)...")
data_sensor = gaussian_forward_UST(image, sensor_x, r_values, lam, 'sensor')
recon_sensor = gaussian_adjoint_UST(data_sensor, image.shape, sensor_x, r_values, lam, 'sensor')

print("Calculating Tissue Heterogeneity (Wavy Medium)...")
data_reflector = gaussian_forward_UST(image, sensor_x, r_values, lam, 'reflector')
recon_reflector = gaussian_adjoint_UST(data_reflector, image.shape, sensor_x, r_values, lam, 'reflector')

print("Calculating Triangle Jitter (Jagged Surface)...")
data_triangle = gaussian_forward_UST(image, sensor_x, r_values, lam=2.0, distortion_type='triangle')
recon_triangle = gaussian_adjoint_UST(data_triangle, image.shape, sensor_x, r_values, lam=2.0, distortion_type='triangle')

# 4. 畫圖比較
# fig, axs = plt.subplots(2, 3, figsize=(14, 8))
fig, axs = plt.subplots(3, 2, figsize=(14, 8))
# ==========================================

# 第一排：小 Lambda (寬頻/模糊)
# axs[0,0].imshow(image, cmap='gray'); axs[0,0].set_title(r"Object $u(x)$")
# axs[0,0].xaxis.tick_top(); axs[0,0].xaxis.set_label_position('top')

# axs[0,1].imshow(data_small, cmap='gray', aspect='auto')
# axs[0,1].set_title(fr"Data $Ru(x_0, r)$, $\lambda={lam_small}$ (Fat)")
# axs[0,1].xaxis.tick_top(); axs[0,1].xaxis.set_label_position('top')

# axs[0,2].imshow(recon_small, cmap='gray')
# axs[0,2].set_title(fr"Recon $R^*$, $\lambda={lam_small}$")
# axs[0,2].xaxis.tick_top(); axs[0,2].xaxis.set_label_position('top')

# # 第二排：大 Lambda (高頻/銳利)
# axs[1,0].imshow(image, cmap='gray'); axs[1,0].set_title(r"Object $u(x)$")
# axs[1,0].xaxis.tick_top(); axs[1,0].xaxis.set_label_position('top')

# axs[1,1].imshow(data_large, cmap='gray', aspect='auto')
# axs[1,1].set_title(fr"Data $Ru(x_0, r)$, $\lambda={lam_large}$ (Sharp)")
# axs[1,1].xaxis.tick_top(); axs[1,1].xaxis.set_label_position('top')

# axs[1,2].imshow(recon_large, cmap='gray')
# axs[1,2].set_title(fr"Recon $R^*$, $\lambda={lam_large}$")
# axs[1,2].xaxis.tick_top(); axs[1,2].xaxis.set_label_position('top')

# ==========================================

# 1. 理想狀態
axs[0,0].imshow(data_ideal, cmap='gray', aspect='auto')
axs[0,0].set_title(r"Ideal Data $Ru$ (Perfect Hyperbola)")
axs[0,1].imshow(recon_ideal, cmap='gray')
axs[0,1].set_title(r"Ideal Recon $R^*u$")

# 2. 皮膚擾動
axs[1,0].imshow(data_sensor, cmap='gray', aspect='auto')
axs[1,0].set_title(r"Sensor Jitter Data (Wobbly Hyperbola)")
axs[1,1].imshow(recon_sensor, cmap='gray')
axs[1,1].set_title(r"Sensor Jitter Recon")

# 3. 組織擾動
axs[2,0].imshow(data_triangle, cmap='gray', aspect='auto')
axs[2,0].set_title(r"Triangle Wave Distortion")
axs[2,1].imshow(recon_triangle, cmap='gray')
axs[2,1].set_title(r"Triangle Wave Distortion")

plt.tight_layout()
# plt.show()

# ==========================================
# 5. 視覺化隱藏的 Levelset 函數 f_tilde (切片法)
# ==========================================
print("Generating f_tilde Levelset Visualizations...")

# --- 切片 A：固定 y (x0=50, r=60)，看物體空間的波前 ---
# 假設我們選定中間的感測器 (x0=50)，並且看時間 r=60 的那一瞬間
fixed_sx = sensor_x[64] 
fixed_r = 60.0

y_coords, x_coords = np.mgrid[:H, :W]
dists_x = np.sqrt((x_coords - fixed_sx)**2 + y_coords**2)
f_ideal_x = dists_x - fixed_r

# 【套用擾動】這裡傳入的是整張畫布的座標 (x_coords, y_coords)
eps_tissue_x = reflector_perturb(x_coords, y_coords)
f_tilde_x = f_ideal_x + eps_tissue_x

# --- 切片 B：固定 x (x1=50, x2=40)，看數據空間的雙曲線 ---
# 假設我們選定人體深處的一個亮點 (x1=50, x2=40)
fixed_x1, fixed_x2 = 50.0, 40.0

R_grid, X0_grid = np.meshgrid(r_values, sensor_x, indexing='ij')
dists_y = np.sqrt((fixed_x1 - X0_grid)**2 + fixed_x2**2)
f_ideal_y = dists_y - R_grid

# 【套用同一個擾動】注意！這裡傳入的是「固定的」座標 (fixed_x1, fixed_x2)
eps_tissue_y = reflector_perturb(fixed_x1, fixed_x2)
f_tilde_y = f_ideal_y + eps_tissue_y

# ==========================================
# 畫出 Levelset 輪廓圖
# ==========================================
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
cmap = 'RdBu_r'

# 畫切片 A (Object Space)
im1 = axs2[0].imshow(f_tilde_x, cmap=cmap, vmin=-30, vmax=30, extent=[0, W, H, 0])
axs2[0].contour(x_coords, y_coords, f_tilde_x, levels=[0], colors='black', linewidths=2, label=r'Distorted Wavefront ($\tilde{f}=0$)')
axs2[0].plot(fixed_sx, 0, 'r^', markersize=10, label='Sensor') # 標示感測器
axs2[0].legend(loc='lower right', facecolor='white', framealpha=0.8)
axs2[0].set_title(r"$\tilde{f}(y,x)$ w.r.t $x$ (Tissue Het. Wavefront)")
axs2[0].xaxis.tick_top()
fig2.colorbar(im1, ax=axs2[0])

# 畫切片 B (Data Space)
im2 = axs2[1].imshow(f_tilde_y, cmap=cmap, aspect='auto', vmin=-30, vmax=30, extent=[0, W, max_dist, 0])
axs2[1].contour(X0_grid, R_grid, f_ideal_y, levels=[0], colors='gray', linestyles='dashed', linewidths=1, label=r'Ideal Hyperbola ($f=0$)') 
axs2[1].contour(X0_grid, R_grid, f_tilde_y, levels=[0], colors='black', linewidths=2, label=r'Distorted Hyperbola ($\tilde{f}=0$)')
axs2[1].legend(loc='lower right', facecolor='white', framealpha=0.8)
axs2[1].set_title(r"$\tilde{f}(y,x)$ w.r.t $y$ (Tissue Het. Hyperbola)")
axs2[1].xaxis.tick_top()
fig2.colorbar(im2, ax=axs2[1])

plt.tight_layout()

# ==========================================
# 6. 視覺化高斯權重 W = exp(-lambda/2 * f_tilde^2)
# ==========================================
print("Generating Gaussian Weights Visualizations...")

lam_plot = 2.0  # 你可以試著把這裡改成 10.0 或 0.5 來觀察變化

# 核心公式：將算好的 f_tilde 轉換成 0~1 的高斯權重
weight_x = np.exp(-(lam_plot / 2.0) * (f_tilde_x**2))
weight_y = np.exp(-(lam_plot / 2.0) * (f_tilde_y**2))

fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
cmap_weight = 'magma' # magma 色帶非常適合表示能量的強弱 (黑->紫->橘->白)

# --- 畫切片 A (Object Space 的波前能量) ---
im3 = axs3[0].imshow(weight_x, cmap=cmap_weight, vmin=0, vmax=1, extent=[0, W, H, 0])
# 疊加波前中心 (f_tilde=0)
axs3[0].contour(x_coords, y_coords, f_tilde_x, levels=[0], colors='white', linestyles='dashed', linewidths=1.5)
axs3[0].plot(fixed_sx, 0, 'w^', markersize=10, label='Sensor Position') 

# 圖例
axs3[0].plot([], [], color='white', linestyle='dashed', linewidth=1.5, label=r'Wavefront Center ($\tilde{f}=0$)')
axs3[0].legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.6)

axs3[0].set_title(fr"Gaussian Weights $w.r.t$ $x$ ($\lambda={lam_plot}$)")
axs3[0].xaxis.tick_top()
fig3.colorbar(im3, ax=axs3[0])


# --- 畫切片 B (Data Space 的雙曲線能量) ---
im4 = axs3[1].imshow(weight_y, cmap=cmap_weight, aspect='auto', vmin=0, vmax=1, extent=[0, W, max_dist, 0])
# 疊加雙曲線中心 (f_tilde=0)
axs3[1].contour(X0_grid, R_grid, f_tilde_y, levels=[0], colors='red', linestyles='dashed', linewidths=1.5)

# 圖例
axs3[1].plot([], [], color='red', linestyle='dashed', linewidth=1.5, label=r'Hyperbola Center ($\tilde{f}=0$)')
axs3[1].legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white', framealpha=0.6)

axs3[1].set_title(fr"Gaussian Weights $w.r.t$ $y$ ($\lambda={lam_plot}$)")
axs3[1].xaxis.tick_top()
fig3.colorbar(im4, ax=axs3[1])

plt.tight_layout()
plt.show()
