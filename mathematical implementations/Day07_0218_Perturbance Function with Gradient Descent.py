import numpy as np
import matplotlib.pyplot as plt

print("--- Learning Epsilon ---")

x_eps = np.linspace(0, 1, 200).reshape(-1, 1)
noise = 0.5
true_epsilon = 2.0 * np.sin(2 * np.pi * x_eps) - 1.5 * np.cos(4 * np.pi * x_eps) + noise * np.random.randn(200, 1)

# 特徵轉換矩陣 (Fourier Basis): cos(0), cos(2pi x), sin(2pi x), cos(4pi x), sin(4pi x)...
def create_fourier_features(x, J):
    features = [np.ones_like(x)] # cos(0) = 1
    for j in range(1, J + 1):
        features.append(np.cos(2 * np.pi * j * x))
        features.append(np.sin(2 * np.pi * j * x))
    return np.hstack(features) # 把這些特徵合併成一個大矩陣
# works as the fixed input X as keeping the cos/sin creates non-linearity

J = 3 # arbitary
Phi = create_fourier_features(x_eps, J)
num_features = Phi.shape[1]

# 3. Gradient Descent
alpha_guess = np.random.randn(num_features, 1) # random initial alpha
s_eps = 0.5   # step size
K_eps = 500   # iter count

loss_record = []

for k in range(K_eps):
    # epsilon = Phi * alpha
    eps_pred = Phi @ alpha_guess
    error = eps_pred - true_epsilon
    
    loss = (1 / (2 * len(x_eps))) * np.sum(error**2)
    loss_record.append(loss)
    gradient = (1 / len(x_eps)) * (Phi.T @ error)
    
    # Update
    alpha_guess = alpha_guess - s_eps * gradient

# visualizations
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(x_eps.flatten(), true_epsilon.flatten(), color='gray', label='True Distorted $\epsilon$', s=10, alpha=0.5)
axs[0].plot(x_eps.flatten(), (Phi @ alpha_guess).flatten(), 'r-', linewidth=2, label=r'Learned $\epsilon(x, \vec{\alpha})$')

axs[0].set_title(r"Learning Perturbation $\epsilon$")
axs[0].legend()

# Task 2: Loss Curve
axs[1].plot(loss_record, 'b-', linewidth=2)
axs[1].set_title("Loss Curve")
axs[1].set_xlabel("Epoch k"); axs[1].set_ylabel(r"Loss $\mathcal{L}(\vec{\alpha})$")

axs[0].grid(True, linestyle='--', alpha=0.6)
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
