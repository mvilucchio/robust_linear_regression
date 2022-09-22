from src.numerics import (
    _loss_and_gradient_double_quad,
    data_generation,
    measure_gen_decorrelated,
    measure_gen_double,
)
import numpy as np
import matplotlib.pyplot as plt

N_SAMPLES = 100
N_POINTS = 5
RANGE = 3
SMALL_RANGE = 0.5
REG_PARAM = 0.5
A = 1.0

if __name__ == "__main__":

    # np.random.seed(3)

    ax = plt.figure().add_subplot(projection="3d")

    xs, ys, _, _, theta_0_teacher = data_generation(
        measure_gen_decorrelated,  # measure_gen_double,
        2,
        N_SAMPLES,
        1,
        {"delta_small": 0.1, "delta_large": 2.0, "percentage": 0.1, "beta": 1.0},
    )

    print("true theta: ", theta_0_teacher)

    xs_norm = np.divide(xs, np.sqrt(2))

    # W_X, W_Y = np.meshgrid(
    #     np.linspace(-RANGE, RANGE, N_POINTS), np.linspace(-RANGE, RANGE, N_POINTS)
    # )

    W_X, W_Y = np.meshgrid(
        np.linspace(
            theta_0_teacher[0] - SMALL_RANGE, theta_0_teacher[0] + SMALL_RANGE, N_POINTS
        ),
        np.linspace(
            theta_0_teacher[1] - SMALL_RANGE, theta_0_teacher[1] + SMALL_RANGE, N_POINTS
        ),
    )

    loss_value = np.empty_like(W_X)

    iterator = range(N_POINTS)

    for idx_x, idx_y in zip(iterator, iterator):
        if idx_x == idx_y:
            print(idx_x, idx_y)
            loss_value[idx_x, idx_y] = 0
        else:
            loss_value[idx_x, idx_y], _ = _loss_and_gradient_double_quad(
                np.append(W_X[idx_x, idx_y], W_Y[idx_x, idx_y]),
                xs_norm,
                ys,
                REG_PARAM,
                1.0,
            )

    ax.plot_surface(W_X, W_Y, loss_value, antialiased=False)
    # ax.scatter(theta_0_teacher[0], theta_0_teacher[1], [0.0], color="r")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()
