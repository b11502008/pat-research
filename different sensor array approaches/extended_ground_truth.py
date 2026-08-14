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

def generate_complex_u(key):
    keys = jr.split(key, 8)
    y_coords, x_coords = jnp.mgrid[:H, :W]

    u_delta = jnp.zeros(H * W)
    indices = jr.randint(keys[0], (15,), 0, H * W)
    u_delta = u_delta.at[indices].set(1.0).reshape(H, W)

    fx = jr.uniform(keys[1], minval=0.2, maxval=0.6)
    fy = jr.uniform(keys[2], minval=0.2, maxval=0.6)
    u_checker = jnp.where(jnp.sin(fx * x_coords) * jnp.sin(fy * y_coords) > 0, 1.0, 0.0)

    def make_disk(k):
        subkeys = jr.split(k, 3)
        cy = jr.randint(subkeys[0], (), 10, H - 10)
        cx = jr.randint(subkeys[1], (), 10, W - 10)
        radius = jr.randint(subkeys[2], (), 3, 10) # 圓變小，但數量變多
        return jnp.where((x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2, 1.0, 0.0)
    
    disk_keys = jr.split(keys[3], 4) # 產生 4 個圓
    u_disks = jax.vmap(make_disk)(disk_keys).sum(axis=0)
    u_disks = jnp.clip(u_disks, 0.0, 1.0)

    cross_cx = jr.randint(keys[4], (), 15, W - 15)
    cross_cy = jr.randint(keys[5], (), 15, H - 15)
    u_cross = jnp.where((jnp.abs(x_coords - cross_cx) < 3) | (jnp.abs(y_coords - cross_cy) < 3), 1.0, 0.0)

    u_noise = jr.normal(keys[6], (H, W))

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
H, W = 80, 80
lam = 30 # 💥 固定 Lambda，抓取幾何細節
J = 20
extend_ratio = 1.2  # 左右各往外延伸 1 倍寬度 (例如 W=128，感測器會從 -128 排到 256)
n_sensors = 350     # 陣列變寬了，感測器數量也要增加 (原本 128 -> 256)

# 💥 2. 產生新的感測器座標
sensor_x = jnp.linspace(-extend_ratio * W, W + extend_ratio * W, n_sensors)
# n_sensors = 128
# sensor_x = jnp.linspace(0, W, n_sensors)
# max_dist = int(jnp.sqrt(H**2 + W**2))
max_x_dist = (1.0 + extend_ratio) * W
max_dist = int(jnp.sqrt(H**2 + max_x_dist**2)) + 5
sensor_span = W + 2.0 * extend_ratio * W
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
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2 + 1e-8) - y[1]
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
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2 + 1e-8) - y[1] 

        self.sensor_mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=16, 
                                     depth=2, activation=jnp.sin, key=key) 

    def __call__(self, y, u, X_0, lambda_val):
        sx, r = y[0], y[1]
        sx_norm = jnp.array([sx / sensor_span])
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
lr_schedule = optax.cosine_decay_schedule(init_value=2e-4, decay_steps=EPOCHS)
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
    sx_norm = jnp.array([sx / sensor_span])
    raw_eps = R_L.sensor_mlp(sx_norm)[0]
    return 2.0 * jax.nn.tanh(raw_eps)
eps_learned_vals = jax.vmap(get_learned_eps)(sensor_x)

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
def neumann_reconstruct_calib(shift_fn, D_data, lam_for_inverse, k_folds=10, alpha=6e-3):
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
    sx_norm = jnp.array([sx / sensor_span])
    raw_eps = R_L.sensor_mlp(sx_norm)[0]
    return 2.0 * jax.nn.tanh(raw_eps)
def shift_gt(sx): 
    return R_GT.epsilon(sx)

# 進行影像重建 (使用跟訓練時一樣的lam以獲得最高解析度)
# 將 alpha 調小(如 5e-4)，讓 Neumann Series 銳化效果更清晰。
u_recon_gt = neumann_reconstruct_calib(shift_gt, d_gt_test, lam_for_inverse=lam, k_folds=10, alpha=5e-4)

# ==========================================
# 📐 計算原圖與 Ideal Reconstruction 的量化指標 (調參基準線)
# ==========================================
# skimage 需要 2D 的 numpy 陣列來計算結構相似度
u_test_2d = np.array(u_test.reshape(H, W))
u_recon_gt_np = np.array(u_recon_gt)

# 計算完美物理重建與真實原圖的差距
psnr_ideal = compute_psnr(u_test_2d, u_recon_gt_np, data_range=1.0)
ssim_ideal = compute_ssim(u_test_2d, u_recon_gt_np, data_range=1.0)

print("==========================================")
print(f"🌟 [Ideal Baseline] PSNR: {psnr_ideal:.2f} dB | SSIM: {ssim_ideal:.4f}")
print("==========================================")

# # ==========================================
# # 📸 繪圖：原圖 vs 完美重建 (調參視覺化)
# # ==========================================
# fig_baseline, axs_base = plt.subplots(1, 2, figsize=(10, 5))

# # 1. 原始圖片 (Ground Truth Object)
# axs_base[0].imshow(u_test_2d, cmap='gray', vmin=0.0, vmax=1.0)
# axs_base[0].set_title("Original Object\n(Ground Truth)", fontsize=16, fontweight='bold')
# axs_base[0].axis('off')

# # 2. 理論完美重建 (Ideal Reconstruction)
# title_ideal = (rf"Ideal Reconstruction" + 
#                f"\nPSNR: {psnr_ideal:.2f} dB | SSIM: {ssim_ideal:.4f}")
# axs_base[1].imshow(u_recon_gt_np, cmap='gray', vmin=0.0, vmax=1.0)
# axs_base[1].set_title(title_ideal, fontsize=16, fontweight='bold', color='darkblue')
# axs_base[1].axis('off')

# plt.tight_layout()
# plt.savefig('tuning_baseline.png', dpi=300, bbox_inches='tight')
# plt.show()
