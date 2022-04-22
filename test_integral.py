import src.numerical_functions as numfun
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

m, q, sigma = 0.9, 0.99, 0.9
delta_small, delta_large, eps = 0.1, 10.0, 0.3
delta = 1.0
a = 0.5

int_m = numfun.m_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
int_q = numfun.q_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
int_sigma = numfun.sigma_hat_equation_Huber_single_noise(m, q, sigma, delta, a)

print("m : {} q : {} sigma : {}".format(int_m, int_q, int_sigma))

borders = numfun.find_integration_borders_square(
    lambda y, xi: numfun.m_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
    np.sqrt((1 + delta)),
    1.0,
)

mb = borders[0][1]

x = np.linspace(-mb, mb, 200)
y = np.linspace(-mb, mb, 200)

X, Y = np.meshgrid(x, y)
Z = np.empty_like(X)

for idx, xx in enumerate(x):
    for jdx, yy in enumerate(y):
        Z[idx, jdx] = numfun.sigma_integral_Huber_single_noise(
            yy, xx, q, m, sigma, delta, a
        )

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, Z)
ax.plot(x, numfun.border_plus_Huber(x, m, q, sigma, a), label="upper")
ax.plot(x, numfun.border_minus_Huber(x, m, q, sigma, a), label="lower")
ax.legend()

print(mb)

print("dbl")
print(
    dblquad(
        numfun.m_integral_Huber_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta, a),
    )[0]
)

print(
    dblquad(
        numfun.q_integral_Huber_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta, a),
    )[0]
)

print(
    dblquad(
        numfun.sigma_integral_Huber_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta, a),
    )[0]
)

plt.show()
