import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from src.numerical_functions import q_integral_BO_decorrelated_noise

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.05)
Y = np.arange(-5, 5, 0.05)
XX, YY = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
Z = np.empty_like(XX)

for idx, x in enumerate(X):
    for jdx, y in enumerate(Y):
        Z[idx, jdx] = -10

delta_small = 0.1
delta_large = 0.01
eps = 0.3
beta = 0.0
# 0.21488070233335826 0.8483598448049128 0.9178670890622806
# 0.18020023661671264 0.8797802722115708 0.34253488286262723
q = 0.99
m = 0.21488070233335826
sigma = 0.9178670890622806

for idx, x in enumerate(X):
    for jdx, y in enumerate(Y):
        Z[idx, jdx] = q_integral_BO_decorrelated_noise(
            y, x, q, m, sigma, delta_small, delta_large, eps, beta
        )

# Plot the surface.
surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter("{x:.02f}")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
