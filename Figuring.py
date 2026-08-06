import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 參數設定
# -------------------------
x1, x2 = 0.3, 0.5      # object point
s0, r0 = 0.2, 0.7      # data point (radius now)

s = np.linspace(0, 1, 800)
theta = np.linspace(0, np.pi, 800)   # 上半圓

color_obj = "tab:green"
color_data = "tab:blue"

# 統一顯示範圍
xmin, xmax = 0, 1
ymin, ymax = 0, 1

# -------------------------
# 建立圖形
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# =========================
# 左：Object Space
# =========================
axs[0].set_title("Object Space X")
axs[0].set_xlabel("$x_1$")
axs[0].set_ylabel("$x_2$")

# 半圓
x1_circle = s0 + r0*np.cos(theta)
x2_circle = r0*np.sin(theta)
axs[0].plot(x1_circle, x2_circle, color=color_data, linewidth=2, label="Isodelay Curve")

# sensor surface
axs[0].axhline(0, linestyle='--', linewidth=1)

# object point
axs[0].scatter(x1, x2, s=100, color=color_obj, label="Object point")
axs[0].scatter(s0, 0, marker='x', s=100, color=color_data, label="Sensor")

axs[0].set_xlim(xmin, xmax)
axs[0].set_ylim(ymin, ymax)
axs[0].invert_yaxis()
axs[0].legend(loc="lower right")
axs[0].set_aspect('equal', adjustable='box')

# =========================
# 右：Data Space
# =========================
axs[1].set_title("Data Space Y")
axs[1].set_xlabel("sensor position $s$")
axs[1].set_ylabel("radius $r$")

# hyperbola
r = np.sqrt((x1 - s)**2 + x2**2)
axs[1].plot(s, r, color=color_obj, linewidth=2, label="Travel-Time Curve")

# measurement point
axs[1].scatter(s0, r0, s=100, color=color_data, label="Data point")
axs[1].scatter(x1, x2, s=100, color=color_obj, label="Nearest point")
axs[0].scatter(s0, 0, marker='x', s=100, color=color_data, label="Sensor")

axs[1].set_xlim(xmin, xmax)
axs[1].set_ylim(ymin, ymax)
axs[1].invert_yaxis()
axs[1].legend(loc="lower right")
axs[1].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("geometry_duality_equal_scale.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# x 軸
x = np.linspace(-1, 1, 1000)

# 定義 lambda 逼近 delta
def lambda_delta(x, lambd):
    return np.sqrt(lambd/(2*np.pi)) * np.exp(-lambd/2 * x**2)

# 不同 lambda 值，lambda 越大函數越尖
lambdas = [5, 20, 100, 500]

plt.figure(figsize=(8,5))
for l in lambdas:
    plt.plot(x, lambda_delta(x, l), label=f'$\lambda$={l}')

plt.arrow(0, 0, 0, 9, width=0.01, head_width=0.05, head_length=1, fc='purple', ec='purple', label=f'$\lambda=\infty$')

plt.title(f"Gaussian Approaching $\delta(x)$")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# x 軸
x = np.linspace(-0.5, 0.5, 1000)
y = np.linspace(0, 20, 1000)

# 理想 Dirac delta 的示意：用箭頭表示
plt.figure(figsize=(8,5))

# 畫箭頭在 x=0
plt.arrow(0, 0, 0, 19, width=0.01, head_width=0.05, head_length=1, fc='r', ec='r', label='Dirac delta')

plt.xlim(-0.5, 0.5)
plt.ylim(0, 20)
plt.title("Ideal Dirac Delta Function")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.grid(True)

# 標註 δ(x)
plt.text(0.05, 1.05, f"$\delta(x)$", color='r', fontsize=12)

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

n_sensors = 10
sx = np.linspace(0, 2, n_sensors)
perturbed_sx = sx - np.tanh((sx - 0.5) / 10.0)

plt.figure(figsize=(4,4))

# title, labels
plt.title("Ideal & Realistic Sensor Locations")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(0,1)
plt.ylim(-0.2, 0.8)

# scatter plot
plt.plot(np.linspace(0, 2, 100), np.zeros(100), 'g--', label="Surface")
plt.scatter(perturbed_sx, np.zeros_like(perturbed_sx), marker='x', color="red", label="Displaced Sensors")
plt.scatter(sx, np.zeros_like(sx), marker='.', color="blue", label="Ideally Located Sensors")

# invert y-axis
plt.gca().invert_yaxis()

# keep aspect ratio equal
plt.gca().set_aspect('equal', adjustable='box')

plt.legend(loc="lower right")
plt.show()