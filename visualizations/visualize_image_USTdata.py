from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax

class UST(eqx.Module):
    f: Callable
    epsilon: Callable

    def __init__(self):
        self.f = lambda y, x: jnp.sqrt((x[0] - y[0])**2 + x[1]**2) - y[1]
        self.epsilon = lambda y, x: 5.0 * jnp.sin(0.1 * y[0])
        # fixed f(y, x) and epsilon(y, x)

    def __call__(self, y, u, X_0, lambda_val):
        L_y = jax.vmap(self.f, in_axes=(None, 0))(y, X_0) + jax.vmap(self.epsilon, in_axes=(None, 0))(y, X_0)
        K_y = jnp.exp(-(lambda_val / 2.0) * (L_y ** 2))
        C = jnp.sqrt(lambda_val / (2.0 * jnp.pi))        
        return C * jnp.mean(K_y * u) # The total sound amplitude received on the circle
        # whenever received a set of data, compute Ru
H, W = 100, 100
lam = 5.0

image = jnp.zeros((H, W))
image = image.at[30, 70].set(1.0)
image = image.at[70, 30].set(1.0)
u_signal = image.flatten()
# image u definition

y_coords, x_coords = jnp.mgrid[:H, :W] # from image to meshgrid
X_grid = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1) # stack for NN to take in
# till here is still the object only

# here we define the sensors
n_sensors = 128
sensor_x = jnp.linspace(0, W, n_sensors)
max_dist = int(jnp.sqrt(H**2 + W**2)) # maximum distance to travel = maximum time to travel bcs speed of sound is fixed
r_values = jnp.arange(max_dist, dtype=jnp.float32)

ust_model = UST() #initialize UST Transformer

def compute_single_pixel(sx, r):
    return ust_model(jnp.array([sx, r]), u_signal, X_grid, lam)
    # for every sensor and the time of sound travel, compute that corresponding Ru

@jax.jit
def compute_one_sensor(sx):
    return jax.vmap(compute_single_pixel, in_axes=(None, 0))(sx, r_values)
    # fix sx, feed r_values --> for every sensor at sx, its reflective pixel for all r (time)

# "for each sensor" is calculated, so here we map forward all sensors
#  (n_sensors, num_r) --> .T --> (num_r, n_sensors)
data_space_jax = lax.map(compute_one_sensor, sensor_x).T
# lax.map is slightly slower than vmap, but takes on less memory

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original Object $u(x)$")
axs[0].xaxis.tick_top()
axs[0].xaxis.set_label_position('top')

axs[1].imshow(data_space_jax, cmap='gray', aspect='auto')
axs[1].set_title(rf"UST Data Space (Sensor Jitter, $\lambda={lam}$)")
axs[1].xaxis.tick_top()
axs[1].xaxis.set_label_position('top')

plt.tight_layout()
plt.savefig('try03.png', dpi=300)
