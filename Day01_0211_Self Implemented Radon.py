import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 拆分函數一：單純負責「算出 Sinogram」
# ==========================================
def forward_radon(image, steps=180):
    """
    image: gray scale img as a 2D array
    steps: how many angles to scan
    """
    sinogram = []
    for angle in range(steps):
        rotated_img = rotate(image, angle, reshape=False, mode='constant', cval=0)
        projection = np.sum(rotated_img, axis=0)
        sinogram.append(projection)
        
    return np.array(sinogram).T

# ==========================================
# 2. 拆分函數二：專門負責「建立 R 矩陣」
# ==========================================
def build_R_matrix(H, W, steps):
    num_pixels = H * W
    num_msrs = W * steps
    R = np.zeros((num_msrs, num_pixels))
    
    # 用迴圈在每一個像素點放一個 "1" (Delta function)，然後算它的影子
    for k in range(num_pixels):
        img = np.zeros((H, W))
        i = k // W
        j = k % W
        img[i, j] = 1.0
        
        # 呼叫純淨版的 forward_radon，打破無限迴圈！
        sino = forward_radon(img, steps=steps)
        R[:, k] = sino.flatten()
        
    return R

def efficient_neumann(A, k_folds):
    """
    Fold Approach
    利用倍增法：S_{new} = S_{old} + P * S_{old}
    執行 k 次迴圈，可以得到 2^k 項的總和！
    """
    n = A.shape[0]
    S = np.eye(n)  # 儲存目前的總和
    P = A.copy()   # 儲存目前的乘數 (A, A^2, A^4, A^8...)
    
    for _ in range(k_folds):
        S = S + P @ S  # 提出公因式，直接把總和翻倍
        P = P @ P      # 把乘數平方，準備給下一次用
        
    return S

def trad_neumann(A, num_terms):

    n = A.shape[0]
    S = np.eye(n)
    Ak = A.copy()
    
    for _ in range(num_terms - 1):
        S = S + Ak
        Ak = Ak @ A
        
    return S
    
# ==========================================
# 3. 執行測試 (為了矩陣計算，必須把圖片改小！)
# ==========================================
SIZE = 50  # 原本的 100 太大了，先用 16x16 測試矩陣運算
STEPS = 18 # 掃描 16 個角度

# 建立測試圖片
xs = np.linspace(-1, 1, SIZE, endpoint=True)
X = np.array(np.meshgrid(xs, xs, indexing='ij'))
image = 1.0 * (((X[0]-0.25)**2 + (X[1]-0.25)**2) < 0.125)

print(f"1. 正在計算 {SIZE}x{SIZE} 的 Sinogram...")
my_sinogram = forward_radon(image, steps=STEPS)

print(f"2. 正在建立 R 矩陣 (大小: {SIZE*STEPS} x {SIZE*SIZE})...")
R = build_R_matrix(SIZE, SIZE, steps=STEPS)


# ==========================================
# 4. Neumann Series: Approx. (RR*)^{-1}
# ==========================================
print("3. 開始計算 Neumann Series...")
R_star = R.T
RRt = R @ R_star

# 【數學關鍵】：為了讓 I+A+A^2 收斂，必須確保矩陣的值夠小
# 我們找一個常數 scale，讓 (scale * RRt) 的最大特徵值小於 1
scale = 1.0 / (np.max(np.linalg.eigvals(RRt).real) + 1e-5)
RRt_scaled = scale * RRt

I = np.eye(RRt.shape[0])
A = I - RRt_scaled

start_time = time.time()
inv_RRt_trad = trad_neumann(A, num_terms=1024)
trad_time = time.time() - start_time

print("-> Neumann Series 計算完成！印出反矩陣的左上角 5x5：")
inv_RRt_trad = inv_RRt_trad * scale
print(np.round(inv_RRt_trad[:5, :5], 4))
print(f"Traditional Version Cost: {trad_time:.4f} secs")

start_time = time.time()
inv_RRt_efficient = efficient_neumann(A, k_folds=10)
efficient_time = time.time() - start_time

print("-> Neumann Series 計算完成！印出反矩陣的左上角 5x5：")
inv_RRt_efficient = inv_RRt_efficient * scale
print(np.round(inv_RRt_efficient[:5, :5], 4))
print(f"Fold Approach Version Cost: {efficient_time:.4f} secs")

# ==========================================
# 5. 畫圖驗證
# ==========================================

# ==========================================
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title(f"Original Image ({SIZE}x{SIZE})")
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title("Radon Transform Sinogram")
# plt.imshow(my_sinogram, cmap='gray', aspect='auto')
# plt.xlabel(r"Angle ($\theta$)")
# plt.ylabel(r"Position ($s$)")
# plt.tight_layout()
# plt.show()
# ==========================================

# 1. 計算兩者的絕對誤差
diff_matrix = np.abs(inv_RRt_trad - inv_RRt_efficient)

# 2. 建立畫布 (1列3欄)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- 圖 1：傳統版 ---
# 使用 'viridis' 或是 'RdBu_r' 這種色帶，很適合用來觀察數值矩陣
im1 = ax1.imshow(inv_RRt_trad, cmap='RdBu_r', aspect='auto')
ax1.set_title("Traditional Neumann Inverse")
ax1.set_xlabel("Column Index")
ax1.set_ylabel("Row Index")
fig.colorbar(im1, ax=ax1) # 加上數值色標

# --- 圖 2：高效 Fold 版 ---
im2 = ax2.imshow(inv_RRt_efficient, cmap='RdBu_r', aspect='auto')
ax2.set_title("Fold Approach Inverse")
ax2.set_xlabel("Column Index")
fig.colorbar(im2, ax=ax2)

# --- 圖 3：誤差圖 (Difference) ---
# 使用 'magma' 或 'hot' 色帶，越亮代表誤差越大
im3 = ax3.imshow(diff_matrix, cmap='hot', aspect='auto')
ax3.set_title("Absolute Difference |Trad - Fold|")
ax3.set_xlabel("Column Index")
fig.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()