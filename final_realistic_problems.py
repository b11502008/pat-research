from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax
import jax.random as jr
import optax
import PIL.Image as Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

SEED = 6789
key = jr.PRNGKey(SEED)

# def generate_complex_u(key):
#     keys = jr.split(key, 6)
#     y_coords, x_coords = jnp.mgrid[:H, :W]

#     # 1. Delta Function
#     u_delta = jnp.zeros(H * W)
#     indices = jr.randint(keys[0], (5,), 0, H * W)
#     u_delta = u_delta.at[indices].set(1.0).reshape(H, W)

#     # 2. Step Function
#     step_threshold = step_threshold = jr.randint(keys[1], (), 10, 22)
#     u_step = jnp.where(x_coords > step_threshold, 1.0, 0.0)
    
#     # 3. Square Wave
#     freq = jr.uniform(keys[2], minval=0.1, maxval=0.5)
#     u_square = jnp.where(jnp.sin(freq * y_coords) > 0, 1.0, 0.0)

#     # 4. Ramp Function
#     u_ramp = x_coords / float(W)

#     # 5. Random Noise
#     u_noise = jr.normal(keys[3], (H, W))

#     # 6. Disk image
#     disk_keys = jr.split(keys[4], 3)
#     cy = jr.randint(disk_keys[0], (), 10, H - 10)
#     cx = jr.randint(disk_keys[1], (), 10, W - 10)
#     radius = jr.randint(disk_keys[2], (), 5, 25)
#     u_disk = jnp.where((x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2, 1.0, 0.0)
    
#     # 7. Linear Combination of the functions
#     weights = jr.uniform(keys[4], (6,), minval=0.0, maxval=0.5)   
#     u_combined = (weights[0] * u_delta + 
#                   weights[1] * u_step + 
#                   weights[2] * u_square + 
#                   weights[3] * u_ramp + 
#                   weights[4] * u_noise +
#                   weights[5] * u_disk)
    
#     u_combined = jnp.clip(u_combined, 0.0, 1.0)
#     return u_combined.flatten()

def generate_complex_u(key):
    keys = jr.split(key, 8)
    y_coords, x_coords = jnp.mgrid[:H, :W]

    # 1. 大量離散脈衝點 (Delta) - 增加到 15 點
    u_delta = jnp.zeros(H * W)
    indices = jr.randint(keys[0], (15,), 0, H * W)
    u_delta = u_delta.at[indices].set(1.0).reshape(H, W)

    # 2. 棋盤格/高頻條紋 (Checkerboard) - 創造大量水平與垂直邊緣
    fx = jr.uniform(keys[1], minval=0.2, maxval=0.6)
    fy = jr.uniform(keys[2], minval=0.2, maxval=0.6)
    u_checker = jnp.where(jnp.sin(fx * x_coords) * jnp.sin(fy * y_coords) > 0, 1.0, 0.0)

    # 3. 多重微小圓盤 (Multiple small disks) - 模擬血管或細胞的複雜弧邊
    def make_disk(k):
        subkeys = jr.split(k, 3)
        cy = jr.randint(subkeys[0], (), 10, H - 10)
        cx = jr.randint(subkeys[1], (), 10, W - 10)
        radius = jr.randint(subkeys[2], (), 3, 10) # 圓變小，但數量變多
        return jnp.where((x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2, 1.0, 0.0)
    
    disk_keys = jr.split(keys[3], 4) # 產生 4 個圓
    u_disks = jax.vmap(make_disk)(disk_keys).sum(axis=0)
    u_disks = jnp.clip(u_disks, 0.0, 1.0)

    # 4. 銳利十字交叉 (Sharp Cross) - 提供強烈的直角與角落特徵
    cross_cx = jr.randint(keys[4], (), 15, W - 15)
    cross_cy = jr.randint(keys[5], (), 15, H - 15)
    u_cross = jnp.where((jnp.abs(x_coords - cross_cx) < 3) | (jnp.abs(y_coords - cross_cy) < 3), 1.0, 0.0)

    # 5. 高頻背景雜訊 (Noise)
    u_noise = jr.normal(keys[6], (H, W))

    # 6. 線性疊加所有複雜特徵
    weights = jr.uniform(keys[7], (5,), minval=0.1, maxval=0.5)   
    u_combined = (weights[0] * u_delta + 
                  weights[1] * u_checker + 
                  weights[2] * u_disks + 
                  weights[3] * u_cross + 
                  weights[4] * u_noise)
    
    return jnp.clip(u_combined, 0.0, 1.0).flatten()

# ==========================================
# 🧱 回歸：固定全域物理參數 lambda
# ==========================================
H, W = 75, 75
lam = 28.0 # 💥 固定 Lambda，抓取幾何細節
J = 20
n_sensors = 128
sensor_x = jnp.linspace(0, W, n_sensors)
max_dist = int(jnp.sqrt(H**2 + W**2))
r_values = jnp.arange(max_dist, dtype=jnp.float32)
y_coords, x_coords = jnp.mgrid[:H, :W]
X_grid = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)

keys = jr.split(key, J)
u_batch = jax.vmap(generate_complex_u)(keys)

SCALE_FACTOR = 20.0
C_val = 1.0

# ==========================================
# 🎯 物理 Module Definition
# ==========================================
class GT_UST(eqx.Module):
    # (重要修正) 使用現代 static field 語法
    f: Callable
    epsilon: Callable

    def __init__(self):
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2) - y[1]
        self.epsilon = lambda sx: jnp.tanh((sx - W/2) / 10.0) / C_val 

    def __call__(self, y, u, X_0, lambda_val):
        sx, r = y[0], y[1]
        true_sx = sx + self.epsilon(sx)
        y_true = jnp.array([true_sx, r])
        L_y = jax.vmap(self.f, in_axes=(None, 0))(y_true, X_0)
        K_y = jnp.exp(-(lambda_val / 2.0) * (L_y ** 2))
        numerator = jnp.sum(K_y * u)
        denominator = jnp.sum(K_y) + 1e-8
        return SCALE_FACTOR * (numerator / denominator)

R_GT = GT_UST()

# ==========================================
# 🌊 回歸：固定物理的 Forward 算子
# ==========================================
@eqx.filter_jit
def compute_one_image(model, u_single):
    """純粹利用 model 定義的物理法则(固定lam)進行 Forward"""
    def compute_single_pixel(sx, r):
        # 🎯 使用全域的固定 lam 常數
        return model(jnp.array([sx, r]), u_single, X_grid, lam)
    
    def compute_one_sensor(sx):
        return jax.vmap(compute_single_pixel, in_axes=(None, 0))(sx, r_values)
        
    return lax.map(compute_one_sensor, sensor_x).T

@eqx.filter_jit
def generate_D_GT(u_single):
    """產生固定物理規則下的 GT 數據"""
    return compute_one_image(R_GT, u_single)

# 產生初始訓練數據 batch
D_GT = lax.map(generate_D_GT, u_batch)

class Calibrated_UST(eqx.Module):
    f: Callable
    sensor_mlp: eqx.nn.MLP

    def __init__(self, key):
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2) - y[1] 

        self.sensor_mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=16, 
                                     depth=2, activation=jnp.sin, key=key) 

    def __call__(self, y, u, X_0, lambda_val):
        sx, r = y[0], y[1]
        sx_norm = jnp.array([sx / W])
        raw_eps = self.sensor_mlp(sx_norm)[0]
        eps_pred = 2.0 * jax.nn.tanh(raw_eps)
        y_pred = jnp.array([sx + eps_pred, r])
        L_y = jax.vmap(self.f, in_axes=(None, 0))(y_pred, X_0)
        K_y = jnp.exp(-(lambda_val / 2.0) * (L_y ** 2))
        numerator = jnp.sum(K_y * u)
        denominator = jnp.sum(K_y) + 1e-8
        return SCALE_FACTOR * (numerator / denominator)

R_L = Calibrated_UST(key)

def single_loss(model, u_true, data_GT):
    data_pred = compute_one_image(model, u_true)
    # 使用 Huber Loss 提高最後階段精度
    return jnp.mean(optax.huber_loss(data_pred, data_GT, delta=1.0))

@eqx.filter_value_and_grad
def loss_Grad(model, u_batch, D_GT_batch):
    def loss_fn(args):
        u_single, d_gt_single = args
        return single_loss(model, u_single, d_gt_single)
    batch_losses = lax.map(loss_fn, (u_batch, D_GT_batch))
    return jnp.mean(batch_losses)

# ==========================================
# 優化器與訓練參數 (保留 Cosine Decay)
# ==========================================
EPOCHS = 500
BATCH_SIZE = 20
lr_schedule = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=EPOCHS)
optim = optax.adamw(learning_rate=lr_schedule)
opt_state = optim.init(eqx.filter(R_L, eqx.is_array))

# ==========================================
# 🚀 make_step (回歸：不吃 current_lam)
# ==========================================
@eqx.filter_jit
def make_step(model, state, step_key):
    subkeys = jr.split(step_key, BATCH_SIZE)
    u_b = jax.vmap(generate_complex_u)(subkeys) # new image 
    
    # calculate GT for new images (物理規則固定在 compute_one_image 裡)
    d_gt_b = lax.map(generate_D_GT, u_b)
    
    # Loss, grad & update
    loss_value, grads = loss_Grad(model, u_b, d_gt_b)
    updates, new_state = optim.update(grads, state, eqx.filter(model, eqx.is_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss_value

# ==========================================
# ⏳ 訓練主迴圈 (回歸：固定物理，只學 \epsilon)
# ==========================================
history_loss = []
train_key = jr.PRNGKey(1234)

for epoch in range(1, EPOCHS + 1):
    train_key, step_key = jr.split(train_key)
    
    # 🎯 呼叫簡化版的 make_step
    R_L, opt_state, loss_value = make_step(R_L, opt_state, step_key)
    
    history_loss.append(loss_value.item())
    
    if epoch % 10 == 0 or epoch == 1:
        # 不再印出 Physics Lam，因為它是固定的
        print(f"Epoch {epoch:03d}/{EPOCHS} | Huber Loss: {loss_value:.4f}")

# ==========================================
# 📊 畫出訓練結果：Loss 與 \epsilon(s)
# ==========================================

eps_gt_vals = jax.vmap(R_GT.epsilon)(sensor_x)
def get_learned_eps(sx):
    sx_norm = jnp.array([sx / W])
    raw_eps = R_L.sensor_mlp(sx_norm)[0]
    return 2.0 * jax.nn.tanh(raw_eps)
eps_learned_vals = jax.vmap(get_learned_eps)(sensor_x)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(range(1, EPOCHS + 1), history_loss, color='blue', linewidth=2)
axs[0].set_title("Training Loss Curve", fontsize=14, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=12); axs[0].set_ylabel("Huber Loss", fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.7); axs[0].set_yscale('log')

# (重要修正) 使用 Raw String (r"...") 修復標題的 LaTeX 警告
axs[1].plot(sensor_x, eps_gt_vals, label=f"Ground Truth Shift ($tanh$)", color='green', linestyle='--', linewidth=2.5)
axs[1].plot(sensor_x, eps_learned_vals, label='Learned MLP Shift', color='red', linewidth=2, alpha=0.8)
axs[1].set_title(fr"Sensor Calibration: $\epsilon(s)$", fontsize=14, fontweight='bold')
axs[1].set_xlabel("Sensor Nominal Position ($sx$)", fontsize=12); axs[1].set_ylabel("Position Shift (pixels)", fontsize=12)
axs[1].legend(loc='lower right', fontsize=12); axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].set_ylim(jnp.min(eps_gt_vals) - 0.5, jnp.max(eps_gt_vals) + 0.5)

plt.tight_layout(); plt.savefig('calibration_function_results.png', dpi=300, bbox_inches='tight')

img_orig = Image.open("slp.png")
# 強制縮放到模型的訓練大小 (W x H = 32x32)
img_prepped = img_orig.convert('L').resize((W, H), Image.Resampling.LANCZOS)
u_test_numpy = np.array(img_prepped) / 255.0
u_test_numpy = np.clip(u_test_numpy, 0.0, 1.0)
# 轉為 JAX 陣列並攤平成 1D 陣列
u_test = jnp.array(u_test_numpy).flatten()
print(f"✅ slp.png to ({H}x{W}) ")

d_gt_test = compute_one_image(R_GT, u_test)

@eqx.filter_jit
def neumann_reconstruct_calib(shift_fn, D_data, lam_for_inverse, k_folds=10, alpha=5e-3):
    r_mesh, sx_mesh = jnp.meshgrid(r_values, sensor_x, indexing='ij')
    D_flat = D_data.flatten() 
    eps_array = jax.vmap(shift_fn)(sx_mesh.flatten())
    Y_true = jnp.stack([sx_mesh.flatten() + eps_array, r_mesh.flatten()], axis=-1)

    # 推理時也使用 Module 一致的幾何函數定義
    def compute_f(x, y):
        return jnp.sqrt((x[0] - y[0])**2 + x[1]**2) - y[1]
    
    L_matrix = jax.vmap(jax.vmap(compute_f, in_axes=(None, 0)), in_axes=(0, None))(X_grid, Y_true)
    
    # 💥 核心改變：在測試時，我們可以使用跟訓練時一樣的銳利度。
    K_matrix = jnp.exp(-(lam_for_inverse / 2.0) * (L_matrix ** 2))

    denominator = jnp.sum(K_matrix, axis=0) + 1e-8 
    R_star = SCALE_FACTOR * (K_matrix / denominator) 
    R_op = R_star.T                                   
    M = jnp.dot(R_star, R_op)        
    I_matrix = jnp.eye(H * W) ##
    M_reg = M + alpha * I_matrix
    max_eig_bound = jnp.max(jnp.sum(jnp.abs(M_reg), axis=1))
    scale = 1.0 / (max_eig_bound + 1e-5)
    M_scaled = M_reg * scale
    H_mat = I_matrix - M_scaled
    S = jnp.eye(H * W); P = H_mat ##
    for _ in range(k_folds):
        S = S + jnp.dot(P, S); P = jnp.dot(P, P)
    M_inv = S * scale
    u_bp = jnp.dot(R_star, D_flat) 
    u_recon_flat = jnp.dot(M_inv, u_bp) 
    u_recon_flat = jnp.clip(u_recon_flat, 0.0, 1.0)
    return u_recon_flat.reshape(H, W)

# 定義兩種校正策略 (不需要變)
def shift_uncalibrated(sx): return 0.0
def shift_calibrated(sx):
    sx_norm = jnp.array([sx / W])
    raw_eps = R_L.sensor_mlp(sx_norm)[0]
    return 2.0 * jax.nn.tanh(raw_eps)
def shift_gt(sx): 
    return R_GT.epsilon(sx)

# 進行影像重建 (使用跟訓練時一樣的lam以獲得最高解析度)
# 將 alpha 調小(如 5e-4)，讓 Neumann Series 銳化效果更清晰。
u_test_2d = np.array(u_test.reshape(H, W))
u_recon_gt = neumann_reconstruct_calib(shift_gt, d_gt_test, lam_for_inverse=lam, k_folds=10, alpha=5e-3)
u_recon_uncalib = neumann_reconstruct_calib(shift_uncalibrated, d_gt_test, lam_for_inverse=lam, k_folds=10, alpha=5e-3)
u_recon_calib   = neumann_reconstruct_calib(shift_calibrated,   d_gt_test, lam_for_inverse=lam, k_folds=10, alpha=5e-3)

psnr_uncalib_img = compute_psnr(u_test_2d, u_recon_uncalib, data_range=1.0)
ssim_uncalib_img = compute_ssim(u_test_2d, u_recon_uncalib, data_range=1.0)
psnr_calib_img = compute_psnr(u_test_2d, u_recon_calib, data_range=1.0)
ssim_calib_img = compute_ssim(u_test_2d, u_recon_calib, data_range=1.0)

psnr_uncalib = compute_psnr(u_recon_gt, u_recon_uncalib, data_range=1.0)
ssim_uncalib = compute_ssim(u_recon_gt, u_recon_uncalib, data_range=1.0)
psnr_calib = compute_psnr(u_recon_gt, u_recon_calib, data_range=1.0)
ssim_calib = compute_ssim(u_recon_gt, u_recon_calib, data_range=1.0)

# 印在終端機讓你確認
print(f"📊 To Image:\n PSNR: {psnr_uncalib_img:.2f} → {psnr_calib_img:.2f} dB | SSIM: {ssim_uncalib_img:.4f} → {ssim_calib_img:.4f}")
print(f"📊 To GT:\n PSNR: {psnr_uncalib:.2f} → {psnr_calib:.2f} dB | SSIM: {ssim_uncalib:.4f} → {ssim_calib:.4f}")

labels = ['Uncalibrated\n(Before)', 'AI Calibrated\n(After)']
psnr_values = [psnr_uncalib, psnr_calib]
ssim_values = [ssim_uncalib, ssim_calib]

# 設定海報專用顏色 (暗紅 vs. 亮綠)
colors = ['#d9534f', '#5cb85c']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')

psnr_bottom = max(0, min(psnr_values) - 2.0)
psnr_top = max(psnr_values) + 3.0

# SSIM: 最低分減去 0.05 當底，最高分加上 0.08 留白給文字
ssim_bottom = max(0.0, min(ssim_values) - 0.05)
ssim_top = min(1.15, max(ssim_values) + 0.08)

# --- 左圖：PSNR ---
bars_psnr = ax1.bar(labels, psnr_values, color=colors, edgecolor='black', width=0.4, linewidth=1.5)
ax1.set_ylabel('PSNR (dB) ↑', fontsize=14, fontweight='bold')
ax1.set_title('Reconstruction Accuracy', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylim(psnr_bottom, psnr_top) # 💥 套用動態裁切的 Y 軸
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.tick_params(axis='x', labelsize=13)

# 在柱子上方加上大字體數值
ax1.bar_label(bars_psnr, fmt='%.1f', fontsize=14, fontweight='bold', padding=5)

# --- 右圖：SSIM ---
bars_ssim = ax2.bar(labels, ssim_values, color=colors, edgecolor='black', width=0.4, linewidth=1.5)
ax2.set_ylabel('SSIM Score ↑', fontsize=14, fontweight='bold')
ax2.set_title('Structural Recovery', fontsize=16, fontweight='bold', pad=15)
ax2.set_ylim(ssim_bottom, ssim_top) # 💥 套用動態裁切的 Y 軸
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.tick_params(axis='x', labelsize=13)

# 在柱子上方加上大字體數值
ax2.bar_label(bars_ssim, fmt='%.3f', fontsize=14, fontweight='bold', padding=5)

# --- 儲存與顯示 ---
plt.tight_layout()
plt.savefig('bar_graph.png', dpi=300, bbox_inches='tight')

# ==========================================
# 📸 繪圖：海報專用 1x3 視覺對比 (保留)
# ==========================================
fig4, axs4 = plt.subplots(1, 3, figsize=(16, 5.5))

# 1. 原始 Ground Truth (完美解答)
# 加上對比度歸一化 vmin=0.0, vmax=1.0，展現絕對公平。
axs4[0].imshow(u_recon_gt.reshape(H, W), cmap='gray', vmin=0.0, vmax=1.0)
axs4[0].set_title("Ground Truth Reconstruction", fontsize=16, fontweight='bold')
axs4[0].axis('off')

# 2. 未校正重建 (Uncalibrated)
# 🎯 💥 (重要修正) 使用 Raw String (r"...") 修復標題的 LaTeX 警告
axs4[1].imshow(u_recon_uncalib, cmap='gray', vmin=0.0, vmax=1.0)
axs4[1].set_title(f"Uncalibrated Reconstruction $\lambda={lam}$ \n(Assuming $\epsilon=0$)", fontsize=16, fontweight='bold')
axs4[1].axis('off')

# 3. AI 校正重建 (Calibrated)
# 🎯 💥 (重要修正) 使用 Raw String (r"...") 修復標題的 LaTeX 警告
axs4[2].imshow(u_recon_calib, cmap='gray', vmin=0.0, vmax=1.0)
axs4[2].set_title(f"AI Calibrated Reconstruction\n(Using Learned $\epsilon$)", fontsize=16, fontweight='bold')
axs4[2].axis('off')

plt.tight_layout()
plt.savefig('poster_calibration_comparison.png', dpi=300, bbox_inches='tight')
plt.show()