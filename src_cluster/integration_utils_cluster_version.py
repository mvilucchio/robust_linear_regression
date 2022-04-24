import numpy as np
from scipy.integrate import romb

MULT_INTEGRAL = 10
TOL_INT = 1e-6
N_TEST_POINTS = 300
K_ROMBERG = 10

# if _check_nested_list(square_borders):
#     max_range = square_borders[0][1]
# elif isinstance(square_borders, (int, float)):
#     max_range = square_borders
# else:
#     raise ValueError("square_borders should be either a list of lists or a number.")


def _check_nested_list(nested_list):
    if isinstance(nested_list, list):
        if isinstance(nested_list[0], list):
            if isinstance(nested_list[0][0], (int, float)):
                return True
    return False


def find_integration_borders_square(
    fun, scale1, scale2, mult=MULT_INTEGRAL, tol=TOL_INT, n_points=N_TEST_POINTS
):
    borders = [[-mult * scale1, mult * scale1], [-mult * scale2, mult * scale2]]

    for idx, ax in enumerate(borders):
        for jdx, border in enumerate(ax):

            while True:
                if idx == 0:
                    max_val = np.max(
                        [
                            fun(borders[idx][jdx], pt)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                else:
                    max_val = np.max(
                        [
                            fun(pt, borders[idx][jdx])
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                if max_val > tol:
                    borders[idx][jdx] = borders[idx][jdx] + (
                        -1.0 if jdx == 0 else 1.0
                    ) * (scale1 if idx == 0 else scale2)
                else:
                    break

    for ax in borders:
        ax[0] = -np.max(np.abs(ax))
        ax[1] = np.max(np.abs(ax))

    max_val = np.max([borders[0][1], borders[1][1]])

    borders = [[-max_val, max_val], [-max_val, max_val]]

    return borders


def divide_integration_borders_grid(square_borders, proportion=0.5):  # , sides_square=3
    if proportion >= 1.0 or proportion <= 0.0:
        raise ValueError(
            "proportion should be a number between 0.0 and 1.0 not included."
        )

    max_range = square_borders[0][1]
    mid_range = proportion * max_range

    # 1 | 2 | 3
    # 4 | 5 | 6
    # 7 | 8 | 9

    domain_x = [
        [-mid_range, mid_range],
        [-mid_range, mid_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [-mid_range, mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
    ]

    domain_y = [
        [-mid_range, mid_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [-mid_range, mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-mid_range, mid_range],
        [mid_range, max_range],
    ]

    return domain_x, domain_y


def domains_double_line_constraint(
    square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3
):
    max_range = square_borders[0][1]

    x_test_val = x_fun_upper(max_range, **args3)
    x_test_val_2 = x_fun_upper(-max_range, **args3)  # attention

    if x_test_val > max_range:
        # print("Case 1")
        domain_x = [[-max_range, max_range]] * 3
        domain_y = [
            [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
            [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1),],
            [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
        ]
    elif x_test_val >= 0:
        if x_test_val_2 < -max_range:
            # print("Case 2.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [-max_range, -x_test_val],
                [x_test_val, max_range],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
        else:
            # print("Case 2.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val_2, -x_test_val],
                [x_test_val, -x_test_val_2],
                [-x_test_val, x_test_val],
                [-max_range, x_test_val_2],
                [-x_test_val_2, max_range],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    elif x_test_val > -max_range:
        if x_test_val_2 < -max_range:
            # print("Case 3.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
        else:
            # print("Case 3.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val, -x_test_val],
                [-max_range, x_test_val_2],
                [-x_test_val_2, max_range],
            ]
            domain_y = [
                [lambda x: y_fun_upper(x, **args1), lambda x: max_range],
                [lambda x: -max_range, lambda x: y_fun_lower(x, **args2)],
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    else:
        # print("Case 4")
        domain_x = [[-max_range, max_range]]
        domain_y = [[lambda x: -max_range, lambda x: max_range]]

    return domain_x, domain_y


def domains_double_line_constraint_only_inside(
    square_borders, y_fun_upper, y_fun_lower, x_fun_upper, args1, args2, args3
):
    max_range = square_borders[0][1]

    x_test_val = x_fun_upper(max_range, **args3)
    x_test_val_2 = x_fun_upper(-max_range, **args3)  # attention

    if x_test_val > max_range:
        # print("Case 1")
        domain_x = [[-max_range, max_range]]
        domain_y = [
            [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1),],
        ]
    elif x_test_val >= 0:
        if x_test_val_2 < -max_range:
            # print("Case 2.A")
            domain_x = [
                [-max_range, -x_test_val],
                [x_test_val, max_range],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
        else:
            # print("Case 2.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, -x_test_val],
                [x_test_val, -x_test_val_2],
                [-x_test_val, x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: y_fun_lower(x, **args2), lambda x: y_fun_upper(x, **args1)],
            ]
    elif x_test_val > -max_range:
        if x_test_val_2 < -max_range:
            # print("Case 3.A")
            domain_x = [
                [-max_range, x_test_val],
                [-x_test_val, max_range],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
        else:
            # print("Case 3.B")
            x_test_val_2 = x_fun_upper(-max_range, **args3)
            domain_x = [
                [x_test_val_2, x_test_val],
                [-x_test_val, -x_test_val_2],
                [x_test_val, -x_test_val],
            ]
            domain_y = [
                [lambda x: -max_range, lambda x: y_fun_upper(x, **args1)],
                [lambda x: y_fun_lower(x, **args2), lambda x: max_range],
                [lambda x: -max_range, lambda x: max_range],
            ]
    else:
        # print("Case 4")
        domain_x = [[-max_range, max_range]]
        domain_y = [[lambda x: -max_range, lambda x: max_range]]

    return domain_x, domain_y

def double_romb_integration(fun, x_lower, x_upper, y_lower, y_upper, args=[]):
    x = np.linspace(x_lower, x_upper, 2**K_ROMBERG + 1)
    y = np.linspace(y_lower, y_upper, 2**K_ROMBERG + 1)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)

    F = fun(X, Y, *args)

    return romb(romb(F, dy), dx)