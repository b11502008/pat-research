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

#     u_delta = jnp.zeros(H * W)
#     indices = jr.randint(keys[0], (5,), 0, H * W)
#     u_delta = u_delta.at[indices].set(1.0).reshape(H, W)

#     step_threshold = step_threshold = jr.randint(keys[1], (), 10, 22)
#     u_step = jnp.where(x_coords > step_threshold, 1.0, 0.0)
    
#     freq = jr.uniform(keys[2], minval=0.1, maxval=0.5)
#     u_square = jnp.where(jnp.sin(freq * y_coords) > 0, 1.0, 0.0)

#     u_ramp = x_coords / float(W)
#     u_noise = jr.normal(keys[3], (H, W))

#     disk_keys = jr.split(keys[4], 3)
#     cy = jr.randint(disk_keys[0], (), 10, H - 10)
#     cx = jr.randint(disk_keys[1], (), 10, W - 10)
#     radius = jr.randint(disk_keys[2], (), 5, 25)
#     u_disk = jnp.where((x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2, 1.0, 0.0)
    
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
# 🧱 幾何更新：改為 Circular Array (圓形陣列)
# ==========================================
H, W = 75, 75
# 💥 關鍵修改 1：訓練時使用較低的 lambda (5.0)，防止梯度消失！
train_lam = 28.0 
eval_lam = 28.0 # 測試/重建時使用高 lambda 確保銳利度
J = 20

# 定義圓環參數
R_ring = W * 0.8  # 圓環半徑 (稍微包覆在影像外圍)
center_x, center_y = W / 2.0, H / 2.0

n_sensors = 128
# 感測器參數變為 0 ~ 2*pi 的角度 (theta)
sensor_theta = jnp.linspace(0, 2 * jnp.pi, n_sensors, endpoint=False)

# 算出最遠可能距離 (圓環半徑 + 畫布對角線的一半 + 安全緩衝)
max_dist = int(R_ring + jnp.sqrt((W/2)**2 + (H/2)**2)) + 5
r_values = jnp.arange(max_dist, dtype=jnp.float32)

y_coords, x_coords = jnp.mgrid[:H, :W]
X_grid = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)

keys = jr.split(key, J)
u_batch = jax.vmap(generate_complex_u)(keys)

SCALE_FACTOR = 20.0
C_val = 1.0

# ==========================================
# 🎯 物理 Module Definition (Circular 版)
# ==========================================
class GT_UST(eqx.Module):
    # 💥 必須標記為 static=True 防止 JAX 崩潰
    f: Callable = eqx.field(static=True)
    epsilon: Callable = eqx.field(static=True)

    def __init__(self):
        # 圓周距離公式：先用 theta 算出 (S_x, S_y)，再算與像素 x 的距離
        # 💥 加入 1e-8 防止 jnp.sqrt(0) 產生 NaN 梯度！
        def f_circ(y, x):
            theta, r = y[0], y[1]
            s_x = center_x + R_ring * jnp.cos(theta)
            s_y = center_y + R_ring * jnp.sin(theta)
            return jnp.sqrt((x[0] - s_x)**2 + (x[1] - s_y)**2 + 1e-8) - r
        self.f = f_circ
        
        # 圓環上的位移誤差改成某個平滑的週期函數 (例如 sin 波)
        self.epsilon = lambda theta: 0.1 * jnp.sin(3.0 * theta) / C_val 

    def __call__(self, y, u, X_0, lambda_val):
        theta, r = y[0], y[1]
        true_theta = theta + self.epsilon(theta)
        y_true = jnp.array([true_theta, r])
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
def compute_one_image(model, u_single, current_lam):
    def compute_single_pixel(theta, r):
        return model(jnp.array([theta, r]), u_single, X_grid, current_lam)
    
    def compute_one_sensor(theta):
        return jax.vmap(compute_single_pixel, in_axes=(None, 0))(theta, r_values)
        
    return lax.map(compute_one_sensor, sensor_theta).T

@eqx.filter_jit
def generate_D_GT(u_single):
    # 訓練用數據使用 train_lam
    return compute_one_image(R_GT, u_single, train_lam)

D_GT = lax.map(generate_D_GT, u_batch)

# ==========================================
# 🧠 AI Module (Circular 版)
# ==========================================
class Calibrated_UST(eqx.Module):
    # 💥 必須標記為 static=True 防止 JAX 崩潰
    f: Callable = eqx.field(static=True)
    sensor_mlp: eqx.nn.MLP

    def __init__(self, key):
        def f_circ(y, x):
            theta, r = y[0], y[1]
            s_x = center_x + R_ring * jnp.cos(theta)
            s_y = center_y + R_ring * jnp.sin(theta)
            return jnp.sqrt((x[0] - s_x)**2 + (x[1] - s_y)**2 + 1e-8) - r
        self.f = f_circ

        # 💥 必須將 jnp.sin 包裝，防止 Python 3.13 報錯！
        def activation_sin(x): return jnp.sin(x)

        self.sensor_mlp = eqx.nn.MLP(in_size=1, out_size=1, width_size=16, 
                                     depth=2, activation=activation_sin, key=key) 

    def __call__(self, y, u, X_0, lambda_val):
        theta, r = y[0], y[1]
        
        # 歸一化：因為 theta 範圍是 0 ~ 2*pi，所以除以 2*pi
        theta_norm = jnp.array([theta / (2.0 * jnp.pi)])
        raw_eps = self.sensor_mlp(theta_norm)[0]
        
        # 允許 AI 學到的最大角度偏移 (例如 0.2 弧度)
        eps_pred = 0.2 * jax.nn.tanh(raw_eps) 
        
        y_pred = jnp.array([theta + eps_pred, r])
        L_y = jax.vmap(self.f, in_axes=(None, 0))(y_pred, X_0)
        K_y = jnp.exp(-(lambda_val / 2.0) * (L_y ** 2))
        numerator = jnp.sum(K_y * u)
        denominator = jnp.sum(K_y) + 1e-8
        return SCALE_FACTOR * (numerator / denominator)

R_L = Calibrated_UST(key)

def single_loss(model, u_true, data_GT):
    data_pred = compute_one_image(model, u_true, train_lam)
    return jnp.mean(optax.huber_loss(data_pred, data_GT, delta=1.0))

@eqx.filter_value_and_grad
def loss_Grad(model, u_batch, D_GT_batch):
    def loss_fn(args):
        u_single, d_gt_single = args
        return single_loss(model, u_single, d_gt_single)
    batch_losses = lax.map(loss_fn, (u_batch, D_GT_batch))
    return jnp.mean(batch_losses)

# ==========================================
# 優化器與訓練參數
# ==========================================
EPOCHS = 500  # 稍微增加一點 epochs 確保收斂
BATCH_SIZE = 20
lr_schedule = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=EPOCHS)
optim = optax.adamw(learning_rate=lr_schedule)
opt_state = optim.init(eqx.filter(R_L, eqx.is_array))

# ==========================================
# 🚀 make_step
# ==========================================
@eqx.filter_jit
def make_step(model, state, step_key):
    subkeys = jr.split(step_key, BATCH_SIZE)
    u_b = jax.vmap(generate_complex_u)(subkeys) 
    d_gt_b = lax.map(generate_D_GT, u_b)
    
    loss_value, grads = loss_Grad(model, u_b, d_gt_b)
    updates, new_state = optim.update(grads, state, eqx.filter(model, eqx.is_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss_value

# ==========================================
# ⏳ 訓練主迴圈
# ==========================================
history_loss = []
train_key = jr.PRNGKey(1234)

for epoch in range(1, EPOCHS + 1):
    train_key, step_key = jr.split(train_key)
    R_L, opt_state, loss_value = make_step(R_L, opt_state, step_key)
    history_loss.append(loss_value.item())
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | Huber Loss: {loss_value:.5f}")

# ==========================================
# 📊 畫出訓練結果：Loss 與 \epsilon(theta)
# ==========================================
eps_gt_vals = jax.vmap(R_GT.epsilon)(sensor_theta)
def get_learned_eps(theta):
    theta_norm = jnp.array([theta / (2.0 * jnp.pi)])
    raw_eps = R_L.sensor_mlp(theta_norm)[0]
    return 0.2 * jax.nn.tanh(raw_eps)
eps_learned_vals = jax.vmap(get_learned_eps)(sensor_theta)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(range(1, EPOCHS + 1), history_loss, color='blue', linewidth=2)
axs[0].set_title("Training Loss Curve", fontsize=14, fontweight='bold')
axs[0].set_xlabel("Epoch", fontsize=12); axs[0].set_ylabel("Huber Loss", fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.7); axs[0].set_yscale('log')

axs[1].plot(sensor_theta, eps_gt_vals, label=r"Ground Truth Shift", color='green', linestyle='--', linewidth=2.5)
axs[1].plot(sensor_theta, eps_learned_vals, label='Learned MLP Shift', color='red', linewidth=2, alpha=0.8)
axs[1].set_title(rf"Sensor Calibration: $\epsilon(\theta)$", fontsize=14, fontweight='bold')
axs[1].set_xlabel(r"Sensor Angle $\theta$ (radians)", fontsize=12); axs[1].set_ylabel("Angular Shift (radians)", fontsize=12)
axs[1].legend(loc='lower right', fontsize=12); axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].set_ylim(jnp.min(eps_gt_vals) - 0.05, jnp.max(eps_gt_vals) + 0.05)

plt.tight_layout(); plt.savefig('polar_function_results.png', dpi=300, bbox_inches='tight')

# ==========================================
# 🌌 重建準備與 Neumann Series
# ==========================================
try:
    img_orig = Image.open("slp.png")
    img_prepped = img_orig.convert('L').resize((W, H), Image.Resampling.LANCZOS)
    u_test_numpy = np.array(img_prepped) / 255.0
    u_test_numpy = np.clip(u_test_numpy, 0.0, 1.0)
    u_test = jnp.array(u_test_numpy).flatten()
    print(f"✅ slp.png loaded ({H}x{W})")
except:
    print("❌ 錯誤：找不到 slp.png，提供 Smooth Disk 作為備案。")
    y_g, x_g = jnp.mgrid[:H, :W]; d_s = jnp.sqrt((x_g - W//2)**2 + (y_g - H//2)**2)
    u_test = (1.0 / (1.0 + jnp.exp(1.5 * (d_s - 6.0)))).flatten()

# 💥 測試資料的 GT 使用 eval_lam (32.0) 確保實體邊緣銳利
d_gt_test = compute_one_image(R_GT, u_test, eval_lam)

@eqx.filter_jit
def neumann_reconstruct_calib(shift_fn, D_data, lam_for_inverse, k_folds=10, alpha=5e-3):
    r_mesh, theta_mesh = jnp.meshgrid(r_values, sensor_theta, indexing='ij')
    D_flat = D_data.flatten() 
    eps_array = jax.vmap(shift_fn)(theta_mesh.flatten())
    Y_true = jnp.stack([theta_mesh.flatten() + eps_array, r_mesh.flatten()], axis=-1)

    def compute_f_circ(x, y):
        t, r = y[0], y[1]
        s_x = center_x + R_ring * jnp.cos(t)
        s_y = center_y + R_ring * jnp.sin(t)
        return jnp.sqrt((x[0] - s_x)**2 + (x[1] - s_y)**2 + 1e-8) - r
    
    L_matrix = jax.vmap(jax.vmap(compute_f_circ, in_axes=(None, 0)), in_axes=(0, None))(X_grid, Y_true)
    K_matrix = jnp.exp(-(lam_for_inverse / 2.0) * (L_matrix ** 2))

    denominator = jnp.sum(K_matrix, axis=0) + 1e-8 
    R_star = SCALE_FACTOR * (K_matrix / denominator) 
    R_op = R_star.T                                   
    M = jnp.dot(R_star, R_op)        
    I_matrix = jnp.eye(H * W)
    M_reg = M + alpha * I_matrix
    max_eig_bound = jnp.max(jnp.sum(jnp.abs(M_reg), axis=1))
    scale = 1.0 / (max_eig_bound + 1e-5)
    M_scaled = M_reg * scale
    H_mat = I_matrix - M_scaled
    S = jnp.eye(H * W); P = H_mat
    for _ in range(k_folds):
        S = S + jnp.dot(P, S); P = jnp.dot(P, P)
    M_inv = S * scale
    u_bp = jnp.dot(R_star, D_flat) 
    u_recon_flat = jnp.dot(M_inv, u_bp) 
    u_recon_flat = jnp.clip(u_recon_flat, 0.0, 1.0)
    return u_recon_flat.reshape(H, W)

def shift_uncalibrated(theta): return 0.0
def shift_calibrated(theta):
    theta_norm = jnp.array([theta / (2.0 * jnp.pi)])
    raw_eps = R_L.sensor_mlp(theta_norm)[0]
    return 0.2 * jax.nn.tanh(raw_eps)
def shift_gt(sx): 
    return R_GT.epsilon(sx)

# 進行影像重建 (使用跟訓練時一樣的lam以獲得最高解析度)
# 將 alpha 調小(如 5e-4)，讓 Neumann Series 銳化效果更清晰。
u_test_2d = np.array(u_test.reshape(H, W))
u_recon_gt = neumann_reconstruct_calib(shift_gt, d_gt_test, lam_for_inverse=eval_lam, k_folds=10, alpha=5e-3)
u_recon_uncalib = neumann_reconstruct_calib(shift_uncalibrated, d_gt_test, lam_for_inverse=eval_lam, k_folds=10, alpha=5e-3)
u_recon_calib   = neumann_reconstruct_calib(shift_calibrated,   d_gt_test, lam_for_inverse=eval_lam, k_folds=10, alpha=5e-3)

psnr_uncalib_img = compute_psnr(u_test_2d, u_recon_uncalib, data_range=1.0)
ssim_uncalib_img = compute_ssim(u_test_2d, u_recon_uncalib, data_range=1.0)
psnr_calib_img = compute_psnr(u_test_2d, u_recon_calib, data_range=1.0)
ssim_calib_img = compute_ssim(u_test_2d, u_recon_calib, data_range=1.0)

psnr_uncalib = compute_psnr(u_recon_gt, u_recon_uncalib, data_range=1.0)
ssim_uncalib = compute_ssim(u_recon_gt, u_recon_uncalib, data_range=1.0)

# 計算 AI Calibrated 的分數
psnr_calib = compute_psnr(u_recon_gt, u_recon_calib, data_range=1.0)
ssim_calib = compute_ssim(u_recon_gt, u_recon_calib, data_range=1.0)

# 印在終端機讓你確認
print(f"📊 To Image:\n PSNR: {psnr_uncalib_img:.2f} → {psnr_calib_img:.2f} dB | SSIM: {ssim_uncalib_img:.4f} → {ssim_calib_img:.4f}")
print(f"📊 To GT:\n PSNR: {psnr_uncalib:.2f} → {psnr_calib:.2f} dB | SSIM: {ssim_uncalib:.4f} → {ssim_calib:.4f}")

# ==========================================
# 📸 繪圖：海報專用 1x3 視覺對比
# ==========================================
fig4, axs4 = plt.subplots(1, 3, figsize=(16, 5.5))

axs4[0].imshow(u_recon_gt.reshape(H, W), cmap='gray', vmin=0.0, vmax=1.0)
axs4[0].set_title("Ground Truth Object\n(Target)", fontsize=16, fontweight='bold')
axs4[0].axis('off')

axs4[1].imshow(u_recon_uncalib, cmap='gray', vmin=0.0, vmax=1.0)
axs4[1].set_title(rf"Uncalibrated Reconstruction $\lambda={eval_lam}$" + "\n(Assuming $\epsilon=0$)", fontsize=16, fontweight='bold')
axs4[1].axis('off')

axs4[2].imshow(u_recon_calib, cmap='gray', vmin=0.0, vmax=1.0)
axs4[2].set_title(rf"AI Calibrated Reconstruction" + "\n(Using Learned $\epsilon$)", fontsize=16, fontweight='bold')
axs4[2].axis('off')

plt.tight_layout()
plt.savefig('polar_calibration_comparison.png', dpi=300, bbox_inches='tight')
plt.show()