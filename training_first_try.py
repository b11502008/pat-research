import jax.numpy as jnp
import jax.random as jr
import jax

import equinox as eqx
import optax

import matplotlib.pyplot as plt

LEARNING_RATE = 3e-4 # = step size
EPOCHS = 10000 # = iteration time
SEED = 6789 # random seed, ensure the same noise each time

key = jr.PRNGKey(SEED)
# used to create a pseudorandom number generator (PRNG) key from a given integer seed.
# key is then passed to other jax.random functions to generate REPRODUCIBLE random numbers.

# data creation functions

# ground truth function --> true answer
f_true = lambda x: jnp.cos(2*jnp.pi*x) + jnp.sin(jnp.pi*x)

xs = jnp.linspace(-1, 1, 24) # from -1 to 1, create 24 fixed input X
ys = jax.vmap(f_true)(xs) # calculate the corresponding output Y
# jax.vmap can do it w/o for loop

key, subkey = jr.split(key, 2) # JAX safety trick preventing the creation of repeated random numbers
# key = new mother key; subkey = disposable key
ys_noisy = ys + 0.3*jr.normal(subkey, ys.shape)

# neural network

class Regressor(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jr.split(key, 5) # 4 layers, each needs an independent key to produce random initial weight
        width = 512 # 512 neurons per layer
        self.layers = [
            eqx.nn.Linear('scalar', width, key=keys[0]), # Linear Regression with a scalar input (1 number) and output with 512 numbers
            jax.nn.gelu, # Activation Function that filters positive values
            eqx.nn.Linear(width, width, key=keys[1]), # Linear Regression between layers
            jax.nn.gelu,
            eqx.nn.Linear(width, width, key=keys[2]),
            jax.nn.gelu,
            eqx.nn.Linear(width, 'scalar', key=keys[4]) # output final decision
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    # actually running the process of passing x
    # Linear function is a callable object

key, subkey = jr.split(key, 2)

model = Regressor(subkey) # create and define model



# loss functions and training

def mse( # mean square error
        y, pred_y
):
    return jnp.mean((y-pred_y)**2)

@eqx.filter_jit # Just-In-Time
# XLA compiler traces the Python calculations, compile them into machine language
# save the time in reading Python code line by line by computing machine language
# "filter" out non-number logics
def loss(
        model,
        x,
        y
):
    pred_y = jax.vmap(model)(x) # get prediction from model
    return mse(y, pred_y) # get corresponding loss

optim = optax.adamw(LEARNING_RATE)
# Adam (Adaptive Moment Estimation，自適應動量)
# Learning rate is not constant, Adam remembers the movement history
# if a weight +/- toward a vertain direction constantly, Adam increases this step size
# if it oscillates then otherwise
# W: weight decay --> prevents overfitting, pulls weight toward 0
opt_state = optim.init(eqx.filter(model, eqx.is_array))
# opt_state records the movement history
# filter out the non-array data in model before optimizing

@eqx.filter_jit
def make_step(
        model,
        opt_state,
        x,
        y
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y) # Automatic Differentiation to compute gradient
    # Higher Order Function:
    # 1. 拿 loss 去換一台超級機器
    # super_machine = eqx.filter_value_and_grad(loss) 
    # 2. 把資料丟進超級機器
    # loss_value, grads = super_machine(model, x, y)
    # eqx.filter_value_and_grad 吃的是函數
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    # accord. to new gradients and movement histories, update the status and form a list of adjustments to weights
    model = eqx.apply_updates(model, updates)
    # JAX objects are immutable, hence applying updates means disposing the old one and form a new one
    # b = b - sigma * s
    return model, opt_state, loss_value

for e in range(EPOCHS):
    model, opt_state, loss_value = make_step(model, opt_state, xs, ys_noisy)
    print(f'{loss_value}')



xs_continuous = jnp.linspace(-1, 1, 128)
ys_continuous = jax.vmap(f_true)(xs_continuous) # true answer
ys_continuous_pred = jax.vmap(model)(xs_continuous) # model performance


plt.plot(xs_continuous, ys_continuous,
         color='black', label='true function')
plt.scatter(xs, ys_noisy,
            color='black', label='train points')
plt.plot(xs_continuous, ys_continuous_pred,
         color='red', label='learned function')
plt.legend()
plt.savefig('result.png')
plt.gcf().clear()