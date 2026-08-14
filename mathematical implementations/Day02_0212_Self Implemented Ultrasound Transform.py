import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

# ==========================================
# 🌌 Spatial Softmax Operator 核心生成器
# ==========================================
def build_softmax_operator(image_shape, n_sensors=128, lam=20.0, scale=10.0):
    """
    預先計算 Spatial Softmax 轉移矩陣 A。
    這是一次性成本，算完後 Forward 和 Adjoint 都可以光速執行。
    """
    H, W = image_shape
    sensor_x = np.linspace(0, W, n_sensors)
    sensor_y = 0.0
    max_dist = int(np.sqrt(H**2 + W**2))
    r_values = np.arange(max_dist)

    # 1. 建立 X 空間網格 (像素位置) -> 展平成 (N_pixels, 2)
    y_coords, x_coords = np.mgrid[:H, :W]
    X_grid = np.column_stack([x_coords.flatten(), y_coords.flatten()])
    
    # 2. 建立 Y 空間網格 (感測器 sx 與 時間 r) -> 展平成 (N_Y, 2)
    # 注意：這裡的 indexing='ij' 確保展平後的順序與你的 (max_dist, n_sensors) 容器一致
    r_mesh, sx_mesh = np.meshgrid(r_values, sensor_x, indexing='ij')
    Y_grid = np.column_stack([sx_mesh.flatten(), r_mesh.flatten()])

    # 3. 計算幾何距離誤差 L = 實際物理距離 - 時間距離 r
    sx_col = Y_grid[:, 0:1] # (N_Y, 1)
    r_col  = Y_grid[:, 1:2] # (N_Y, 1)
    x_row  = X_grid[:, 0]   # (N_X,)
    y_row  = X_grid[:, 1]   # (N_X,)

    # 算出距離矩陣 L (Y軸: 所有觀測點, X軸: 所有像素)
    dists = np.sqrt((x_row - sx_col)**2 + (y_row - sensor_y)**2)
    L_matrix = dists - r_col

    # 4. 套用 Spatial Softmax 公式
    K_matrix = np.exp(-(lam / 2.0) * (L_matrix ** 2))
    
    # 對 X 空間 (axis=1) 進行積分歸一化，確保能量守恆
    denominator = np.sum(K_matrix, axis=1, keepdims=True) + 1e-8 
    
    # 乘上權重 w(y)
    A_matrix = scale * (K_matrix / denominator)

    return A_matrix, sensor_x, max_dist

# ==========================================
# 1. Forward Operator (R)
# ==========================================
def forward_R_softmax(image, A_matrix, max_dist, n_sensors):
    """
    實作 R 運算子：D = A @ u
    """
    u_flat = image.flatten()
    
    # 矩陣相乘，完成機率積分
    data_flat = A_matrix @ u_flat
    
    # 變回你的 (r, sx) 容器形狀
    data_space = data_flat.reshape(max_dist, n_sensors)
    return data_space

# ==========================================
# 2. Adjoint Operator (R*)
# ==========================================
def R_star_softmax(data, A_matrix, image_shape):
    """
    實作 R* 運算子：u_recon = A.T @ D
    """
    H, W = image_shape
    data_flat = data.flatten()
    
    # Adjoint 就是核心矩陣的轉置 (Transpose) 乘法！
    u_recon_flat = A_matrix.T @ data_flat
    
    return u_recon_flat.reshape(H, W)

def neumann_reconstruct_softmax(data, A_matrix, image_shape, k_folds=10, alpha=5e-3):
    """
    實作 (R* R)^(-1) R* D，利用倍增法逼近反矩陣。
    """
    H, W = image_shape
    N_pixels = H * W
    data_flat = data.flatten()
    
    print("⏳ 正在計算 R*R 矩陣 (這需要一點時間與記憶體)...")
    # 1. 建立 R* R 矩陣 (M = A^T @ A)
    # A_matrix shape: (N_Y, N_pixels), A_matrix.T shape: (N_pixels, N_Y)
    # M_matrix shape: (N_pixels, N_pixels)
    M_matrix = A_matrix.T @ A_matrix
    
    # 加入 Tikhonov 正規化 (避免除以 0 或奇異矩陣)
    I_matrix = np.eye(N_pixels)
    M_reg = M_matrix + alpha * I_matrix
    
    # 2. 特徵值縮放 (Scaling) 確保收斂
    # 找出矩陣列絕對值總和的最大值，作為最大特徵值的上界
    max_eig_bound = np.max(np.sum(np.abs(M_reg), axis=1))
    scale_factor = 1.0 / (max_eig_bound + 1e-5)
    M_scaled = M_reg * scale_factor
    
    # 3. 初始化倍增法 (Folded Neumann)
    H_mat = I_matrix - M_scaled # 這是殘差矩陣 (Residual)
    S = np.eye(N_pixels)        # 累加器 (S_0 = I)
    P = H_mat                   # 倍增器 (P_0 = M)
    
    print(f"🚀 開始 Folded Neumann 迭代 (k_folds={k_folds})...")
    for k in range(k_folds):
        print(f"   - 迭代 {k+1}/{k_folds} (計算等效展開項數: {2**(k+1)} 項)")
        S = S + P @ S
        P = P @ P
        
    # 4. 完成反矩陣逼近
    M_inv = S * scale_factor
    
    print("✨ 重建影像...")
    # 5. 套用公式：u = M_inv @ (A^T @ D)
    u_bp = A_matrix.T @ data_flat     # 傳統的 Backprojection (R* D)
    u_recon_flat = M_inv @ u_bp       # 乘上反矩陣濾波器
    
    u_recon_flat = np.clip(u_recon_flat, 0.0, 1.0)
    return u_recon_flat.reshape(H, W)

# ------------------------------------------
# 測試與執行區塊
# ------------------------------------------
H, W = 100, 100
R_radius = 6.0
steepness = 1.5
y_grid_test, x_grid_test = np.mgrid[:H, :W]
distances = np.sqrt((x_grid_test - W // 2)**2 + (y_grid_test - H // 2)**2)
image = 1.0 / (1.0 + np.exp(steepness * (distances - R_radius)))

# 加入一點小雜點讓重建的銳利度對比更明顯
image[30:35, 70:75] = 1.0
image[70:75, 30:35] = 1.0

# 1. 建立算子
A_mat, s_x, m_dist = build_softmax_operator((H, W), n_sensors=128, lam=10.0, scale=1.0)

# 2. 模擬機器收集數據
data_space = forward_R_softmax(image, A_mat, m_dist, n_sensors=128)

# 3. 單純反投影 (模糊、有十字假影)
reconstruction_adj = R_star_softmax(data_space, A_mat, (H, W))

# 4. Neumann 反矩陣重建 (銳利、消除假影)
# k_folds=8 相當於展開 2^8 = 256 項，對這張小圖已經非常足夠且速度較快
reconstruction_neu = neumann_reconstruct_softmax(data_space, A_mat, (H, W), k_folds=8, alpha=5e-3)

# ------------------------------------------
# 畫圖驗證
# ------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# 圖 1: Ground Truth
axes[0].set_title(f"Object Space $u(x)$", fontsize=14)
axes[0].imshow(image, cmap='gray')
axes[0].xaxis.tick_top(); axes[0].xaxis.set_label_position('top')

# 圖 2: Sensor Data
data_space_sliced = data_space[:100,:100]
axes[1].set_title(f"Data Space $Ru(x_0, r)$", fontsize=14)
axes[1].imshow(data_space_sliced, cmap='gray', aspect='auto')
axes[1].set_xlabel("Sensor Position")
axes[1].set_ylabel("Radius/Time")
axes[1].xaxis.tick_top(); axes[1].xaxis.set_label_position('top')

# 圖 3: Adjoint (模糊)
axes[2].set_title("Adjoint $R^* D$\n(Blurred & Artifacts)", fontsize=14)
axes[2].imshow(reconstruction_adj, cmap='gray')
axes[2].xaxis.tick_top(); axes[2].xaxis.set_label_position('top')

# 圖 4: Neumann (銳利)
axes[3].set_title("Neumann Inverse $(R^*R)^{-1} R^* D$\n(Sharp & Artifact-free)", fontsize=14)
axes[3].imshow(reconstruction_neu, cmap='gray')
axes[3].xaxis.tick_top(); axes[3].xaxis.set_label_position('top')

plt.tight_layout()
plt.show()
