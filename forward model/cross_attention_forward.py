from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax
import jax.random as jr
import optax
import time

start_time = time.time()
SEED = 6789 # random seed, ensure the same noise each time
key = jr.PRNGKey(SEED)

def generate_complex_u(key):
    keys = jr.split(key, 6)
    y_coords, x_coords = jnp.mgrid[:H, :W]

    # 1. Delta Function
    u_delta = jnp.zeros(H * W)
    indices = jr.randint(keys[0], (5,), 0, H * W)
    u_delta = u_delta.at[indices].set(1.0).reshape(H, W)

    # 2. Step Function
    step_threshold = step_threshold = jr.randint(keys[1], (), 10, 22)
    u_step = jnp.where(x_coords > step_threshold, 1.0, 0.0)
    
    # 3. Square Wave
    freq = jr.uniform(keys[2], minval=0.1, maxval=0.5)
    u_square = jnp.where(jnp.sin(freq * y_coords) > 0, 1.0, 0.0) # when sine > 0: 1; else: 0

    # 4. Ramp Function
    u_ramp = x_coords / float(W)

    # 5. Random Noise
    u_noise = jr.normal(keys[3], (H, W))

    # 6. Disk image
    disk_keys = jr.split(keys[4], 3)
    cy = jr.randint(disk_keys[0], (), 10, H - 10)
    cx = jr.randint(disk_keys[1], (), 10, W - 10)
    radius = jr.randint(disk_keys[2], (), 5, 25)
    u_disk = jnp.where((x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2, 1.0, 0.0)
    
    # 7. Linear Combination of the functions
    weights = jr.uniform(keys[4], (6,), minval=0.0, maxval=0.5)   
    u_combined = (weights[0] * u_delta + 
                  weights[1] * u_step + 
                  weights[2] * u_square + 
                  weights[3] * u_ramp + 
                  weights[4] * u_noise +
                  weights[5] * u_disk)
    
    u_combined = jnp.clip(u_combined, 0.0, 1.0) # clip to restrict the values
    return u_combined.flatten()

H, W = 32, 32
lam = 10.0
J = 20
n_sensors = 128
sensor_x = jnp.linspace(0, W, n_sensors)
max_dist = int(jnp.sqrt(H**2 + W**2)) # maximum distance to travel = maximum time to travel bcs speed of sound is fixed
r_values = jnp.arange(max_dist, dtype=jnp.float32)
y_coords, x_coords = jnp.mgrid[:H, :W] # from image to meshgrid
X_grid = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1) # stack for NN to take in
# till here is still the object only

keys = jr.split(key, J)
u_batch = jax.vmap(generate_complex_u)(keys)
# image u definition

SCALE_FACTOR = 10.0 # to avoid gradient vanishing due to too small loss

class GT_UST(eqx.Module):
    f: Callable
    epsilon: Callable

    def __init__(self):
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2) - y[1]
        self.epsilon = lambda y, x: 5.0 * jnp.sin(0.1 * y[0])
        # fixed f(y, x) and epsilon(y, x)

    def __call__(self, y, u, X_0, lambda_val):
        L_y = jax.vmap(self.f, in_axes=(None, 0))(y, X_0) + jax.vmap(self.epsilon, in_axes=(None, 0))(y, X_0)
        K_y = jnp.exp(-(lambda_val / 2.0) * (L_y ** 2))

        numerator = jnp.sum(K_y * u)
        denominator = jnp.sum(K_y) + 1e-8
        return SCALE_FACTOR * (numerator / denominator)
        # C = jnp.sqrt(lambda_val / (2.0 * jnp.pi))        
        # return SCALE_FACTOR * C * jnp.mean(K_y * u) # The total sound amplitude received on the circle
        # # whenever received a set of data, compute Ru

R_GT = GT_UST()

@eqx.filter_jit
def compute_one_image(model, u_single):
    def compute_single_pixel(sx, r):
        return model(jnp.array([sx, r]), u_single, X_grid, lam)
    
    def compute_one_sensor(sx):
        return jax.vmap(compute_single_pixel, in_axes=(None, 0))(sx, r_values)
        
    return lax.map(compute_one_sensor, sensor_x).T

@eqx.filter_jit
def generate_D_GT(u_single):
    return compute_one_image(R_GT, u_single)

D_GT = lax.map(generate_D_GT, u_batch)
print(f"u_batch shape: {u_batch.shape}  -> (J, length of flattened image)")
print(f"D_GT shape: {D_GT.shape} -> (J, # of r, # of sensor_x)")

class Factorized_UST(eqx.Module):
    f: Callable
    psi_Y_mlp: eqx.nn.MLP
    psi_X_mlp: eqx.nn.MLP

    def __init__(self, key):
        # f(y,x) is not trained
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2) - y[1]
        k1, k2 = jr.split(key, 2)
        
        # jnp.sin as activation
        # 3 hidden layers (depth=3)，64 neurons each (width_size=64)
        
        # 1. \psi_Y: sensor(graph)
        # input (sx, r)，output 64 + 1
        self.psi_Y_mlp = eqx.nn.MLP(in_size=2, out_size=65, width_size=64, 
                                    depth=3, activation=jnp.sin, key=k1)
        
        # 2. \psi_X: object space
        # input (x1, x2)，output 64 as Key Vector
        self.psi_X_mlp = eqx.nn.MLP(in_size=2, out_size=64, width_size=64, 
                                    depth=3, activation=jnp.sin, key=k2)

    def __call__(self, y, u, X_0, lambda_val):
        # Cross-Attention
        # Query
        y_norm = y / 32.0 # size of image
        out_Y = self.psi_Y_mlp(y_norm)
        psi_Y_val = out_Y[:64]  # the first 64 is the query vector
        b_y = out_Y[64] # last is the bias
        
        # Keys
        X_norm = X_0 / 32.0
        psi_X_vals = jax.vmap(self.psi_X_mlp)(X_norm) # (1024, 64)
        
        # (1024, 64) dot (64,) + bias --> 1024 f(y,x)s
        inner_products = jnp.dot(psi_X_vals, psi_Y_val) 
        eps_array = 5.0 * jax.nn.tanh(inner_products + b_y)

        # 原本是 + jax.vmap(self.epsilon)(y, X_0)
        # 現在我們直接把算好的 1024 個擾動值 eps_array 加進去！
        L_y = jax.vmap(self.f, in_axes=(None, 0))(y, X_0) + eps_array
        K_y = jnp.exp(-(lambda_val / 2.0) * (L_y ** 2))
        
        numerator = jnp.sum(K_y * u)
        denominator = jnp.sum(K_y) + 1e-8
        return SCALE_FACTOR * (numerator / denominator)
    
        # C = jnp.sqrt(lambda_val / (2.0 * jnp.pi))        
        # return SCALE_FACTOR * C * jnp.mean(K_y * u)

R_L = Factorized_UST(key)

def single_loss(model, u_true, data_GT):
    data_pred = compute_one_image(model, u_true)
    return jnp.mean((data_pred - data_GT)**2)

@eqx.filter_value_and_grad
def loss_Grad(model, u_batch, D_GT_batch):
    # 寫一個小幫手函數，專門用來吃 lax.map 拆出來的陣列
    def loss_fn(args):
        u_single, d_gt_single = args
        return single_loss(model, u_single, d_gt_single)
    
    # 讓 lax.map 只去拆 u_batch 和 D_GT_batch
    batch_losses = lax.map(loss_fn, (u_batch, D_GT_batch))
    return jnp.mean(batch_losses)

LEARNING_RATE = 3e-4  # 學習率
EPOCHS = 300          # 因為每次都是新題目，我們需要多跑幾圈讓它學
BATCH_SIZE = 20       # 每次餵 20 張新圖

optim = optax.adamw(LEARNING_RATE)
opt_state = optim.init(eqx.filter(R_L, eqx.is_array))

@eqx.filter_jit
def make_step(model, state, step_key):
    subkeys = jr.split(step_key, BATCH_SIZE)
    u_b = jax.vmap(generate_complex_u)(subkeys) # new image 
    
    # calculate GT for new images
    d_gt_b = lax.map(generate_D_GT, u_b)
    
    # Loss, grad & update
    loss_value, grads = loss_Grad(model, u_b, d_gt_b)
    updates, new_state = optim.update(grads, state, eqx.filter(model, eqx.is_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss_value

history_loss = []
train_key = jr.PRNGKey(1234)

for epoch in range(1, EPOCHS + 1):
    train_key, step_key = jr.split(train_key)
    R_L, opt_state, loss_value = make_step(R_L, opt_state, step_key)
    history_loss.append(loss_value.item())
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | MSE Loss: {loss_value:.4f}")

def eps_GT(sx):
    return R_GT.epsilon(jnp.array([sx, 0.0]), jnp.array([0.0, 0.0]))

def eps_L(sx):
    y_norm = jnp.array([sx, 0.0]) / 32.0 # because the GT epsilon only depends on sx, fix r to 0
    out_Y = R_L.psi_Y_mlp(y_norm) # such that psi_Y can take the 2 variables
    psi_Y_val = out_Y[:64]
    b_y = out_Y[64]
    
    x_norm = jnp.array([0.0, 0.0]) / 32.0 # pick random point 0,0 because the GT doesn't depend on it
    psi_X_val = R_L.psi_X_mlp(x_norm)

    return 5.0 * jax.nn.tanh(jnp.dot(psi_X_val, psi_Y_val) + b_y)

eps_gt_vals = jax.vmap(eps_GT)(sensor_x)
eps_learned_vals = jax.vmap(eps_L)(sensor_x)

end_time = time.time()
total_time = end_time - start_time
print(f"total execution time: {total_time} secs")

# fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# axs[0].plot(range(1, EPOCHS + 1), history_loss, color='blue', linewidth=2)
# axs[0].set_title("Training Loss Curve", fontsize=14)
# axs[0].set_xlabel("Epoch", fontsize=12)
# axs[0].set_ylabel("MSE Loss", fontsize=12)
# axs[0].grid(True, linestyle='--', alpha=0.7)

# axs[0].set_yscale('log')

# axs[1].plot(sensor_x, eps_gt_vals, label='Ground Truth Function', 
#             color='green', linestyle='--', linewidth=2.5)
# axs[1].plot(sensor_x, eps_learned_vals, label='Learned MLP', 
#             color='red', linewidth=2, alpha=0.8)

# axs[1].set_title(fr"Comparison of $\epsilon(y, x)$", fontsize=14)
# axs[1].set_xlabel(f"Sensor Position ($sx$)", fontsize=12)
# axs[1].set_ylabel("Perturbation Value", fontsize=12)
# axs[1].legend(loc='upper right', fontsize=12)
# axs[1].grid(True, linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.savefig('training_results.png', dpi=300)



u_test_simple = jnp.zeros((H, W))
u_test_simple = u_test_simple.at[10, 8].set(1.0)
u_test_simple = u_test_simple.at[24, 22].set(1.0)

u_test = u_test_simple.flatten()

d_gt_test = compute_one_image(R_GT, u_test)
d_pred_test = compute_one_image(R_L, u_test)

error_map = jnp.abs(d_gt_test - d_pred_test)

print(f"MSE Loss: {jnp.mean(error_map**2):.6f}")

fig2, axs2 = plt.subplots(1, 4, figsize=(20, 5))

# 1. u(x)
axs2[0].imshow(u_test.reshape(H, W), cmap='gray')
axs2[0].set_title("Original Object (Two Points)", fontsize=14)
axs2[0].axis('off')

# 2. R_GT
img_gt = axs2[1].imshow(d_gt_test, cmap='magma', aspect='auto')
axs2[1].set_title("Ground Truth UST ($D_{GT}$)", fontsize=14)
axs2[1].set_xlabel("Sensor Position")
axs2[1].set_ylabel("Time / Distance ($r$)")
fig2.colorbar(img_gt, ax=axs2[1])

# 3. R_pred
img_pred = axs2[2].imshow(d_pred_test, cmap='magma', aspect='auto')
axs2[2].set_title("AI Predicted UST ($D_{pred}$)", fontsize=14)
axs2[2].set_xlabel("Sensor Position")
fig2.colorbar(img_pred, ax=axs2[2])

# 4. Error Map
img_err = axs2[3].imshow(error_map, cmap='hot', aspect='auto')
axs2[3].set_title("Absolute Error", fontsize=14)
axs2[3].set_xlabel("Sensor Position")
fig2.colorbar(img_err, ax=axs2[3])

plt.tight_layout()
plt.savefig('ultrasound_prediction.png', dpi=300)
