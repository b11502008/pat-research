import numpy as np
import time

def trad_neumann(A, num_terms):

    n = A.shape[0]
    S = np.eye(n)
    Ak = A.copy()
    
    for _ in range(num_terms - 1):
        S = S + Ak
        Ak = Ak @ A
        
    return S

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

# ==========================================
# 測試與驗證
# ==========================================

# 1. 產生一個隨機矩陣 RR* 模擬你的情況 (例如 500x500 的矩陣)
N = 500
np.random.seed(42)
R = np.random.rand(N, N)
RRt = R @ R.T

# 2. 縮放 RR*，確保特徵值小於 1，這樣 Neumann series 才會收斂
scale = 1.0 / (np.max(np.linalg.eigvals(RRt).real) + 0.1)
RRt_scaled = scale * RRt

# 3. 定義 A = I - 縮放後的 RR*
I = np.eye(N)
A = I - RRt_scaled

# ------------------------------------------
# 比較一：傳統版 (要算 1024 項)
# ------------------------------------------
start_time = time.time()
inv_trad = trad_neumann(A, num_terms=1024)
trad_time = time.time() - start_time
print(f"傳統版 (Trad) 運算 1024 項耗時: {trad_time:.4f} 秒")

# ------------------------------------------
# 比較二：高效版 (只要跑 10 次 Fold，因為 2^10 = 1024)
# ------------------------------------------
start_time = time.time()
inv_efficient = efficient_neumann(A, k_folds=10)
efficient_time = time.time() - start_time
print(f"高效版 (Efficient) 運算 1024 項耗時: {efficient_time:.4f} 秒")

# 4. 驗證兩個算出來的結果是不是真的完全一樣？
difference = np.max(np.abs(inv_trad - inv_efficient))
print(f"兩種算法的結果最大誤差: {difference:.2e} (若接近 0 代表結果完全相同)")

# 5. 不要忘記，最後要把 scale 乘回去才是真正的 (RR*) 的反矩陣
final_inverse = inv_efficient * scale
