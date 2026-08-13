import jax.numpy as jnp
import jax.random as jr
import jax

import equinox as eqx
import optax

import matplotlib.pyplot as plt

LEARNING_RATE = 3e-4
EPOCHS = 5000
SEED = 6789

key = jr.PRNGKey(SEED)

# ==========================================
# 修改 1：2D 資料生成 (Data Creation)
# ==========================================

# input (x1, x2); output y
# 3D function: cos(2*pi*x1) * sin(pi*x2)
f_true = lambda x: jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

x1_vals = jnp.linspace(-1, 1, 20) # ranges, size = (1, 20)
x2_vals = jnp.linspace(-1, 1, 20)
X1, X2 = jnp.meshgrid(x1_vals, x2_vals) #meshed into size (20, 20)

# flatten the meshgrid into 400 data with 2 features/numbers (x1, x2) each
# xs.size = (400, 2)
xs = jnp.stack([X1.flatten(), X2.flatten()], axis=1) # input x

# calculated_y + noise = ground true function
ys = jax.vmap(f_true)(xs)
key, subkey = jr.split(key, 2)
ys_noisy = ys + 0.3 * jr.normal(subkey, ys.shape)

# ==========================================
# 修改 2：神經網路第一層 (Neural Network)
# ==========================================

class Regressor(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jr.split(key, 5)
        width = 512
        self.layers = [
            # Input: scalar --> 2 dimensional
            eqx.nn.Linear(2, width, key=keys[0]), 
            jax.nn.gelu,
            eqx.nn.Linear(width, width, key=keys[1]),
            jax.nn.gelu,
            eqx.nn.Linear(width, width, key=keys[2]),
            jax.nn.gelu,
            eqx.nn.Linear(width, 'scalar', key=keys[4]) # output stays the same
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

key, subkey = jr.split(key, 2)
model = Regressor(subkey)

# model training

def mse(y, pred_y):
    return jnp.mean((y-pred_y)**2)

@eqx.filter_jit
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return mse(y, pred_y)

optim = optax.adamw(LEARNING_RATE)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def make_step(model, opt_state, x, y):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

for e in range(EPOCHS):
    model, opt_state, loss_value = make_step(model, opt_state, xs, ys_noisy)
    # if e % 500 == 0 or e == EPOCHS - 1 : # print every 500 epochs for better visualization
    print(f'Epoch {e}: Loss = {loss_value:.4f}')

# a more intense meshgrid for smooth surface
x1_cont = jnp.linspace(-1, 1, 50)
x2_cont = jnp.linspace(-1, 1, 50)
X1_cont, X2_cont = jnp.meshgrid(x1_cont, x2_cont)
xs_cont = jnp.stack([X1_cont.flatten(), X2_cont.flatten()], axis=1)

# preduct the height of meshgrid with model
ys_cont_pred = jax.vmap(model)(xs_cont)
ys_cont_true = jax.vmap(f_true)(xs_cont)

# result is flat, so reshape them back to 2-D
Z_pred = ys_cont_pred.reshape(50, 50)
Z_true = ys_cont_true.reshape(50, 50)

# plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1_cont, X2_cont, Z_pred, cmap='coolwarm', alpha=0.8)
wire = ax.plot_wireframe(X1_cont, X2_cont, Z_true, color='green', alpha=0.4, linewidth=1)

# plot the training data to see model performance
ax.scatter(xs[:, 0], xs[:, 1], ys_noisy, color='black', s=15, label='Train Points')

ax.set_title('Neural Network 2D Prediction')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y (Prediction)')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Predicted Y')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('try01.png', dpi=300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(xs[:, 0], ys_noisy, color='black', alpha=0.4, s=15, label='Train Points')
for i in range(50):
    if i == 0: # 只讓第一條線產生圖例，不然圖例會出現 50 次
        ax1.plot(X1_cont[i, :], Z_pred[i, :], color='red', alpha=0.2, label='Model Performance')
        ax1.plot(X1_cont[i, :], Z_true[i, :], color='black', alpha=0.2, label='Ground Truth Function')
    else:
        ax1.plot(X1_cont[i, :], Z_pred[i, :], color='red', alpha=0.2)
        ax1.plot(X1_cont[i, :], Z_true[i, :], color='black', alpha=0.2)

ax1.set_title('View from X1-Y Plane')
ax1.set_xlabel('X1')
ax1.set_ylabel('Y (Prediction)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.scatter(xs[:, 1], ys_noisy, color='black', alpha=0.4, s=15, label='Train Points')
for i in range(50):
    if i == 0:
        ax2.plot(X2_cont[:, i], Z_pred[:, i], color='blue', alpha=0.2, label='Model Performance')
        ax2.plot(X2_cont[:, i], Z_true[:, i], color='black', alpha=0.2, label='Ground Truth Function')
    else:
        ax2.plot(X2_cont[:, i], Z_pred[:, i], color='blue', alpha=0.2)
        ax2.plot(X2_cont[:, i], Z_true[:, i], color='black', alpha=0.2)

ax2.set_title('View from X2-Y Plane')
ax2.set_xlabel('X2')
ax2.set_ylabel('Y (Height)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('try02.png', dpi=300)