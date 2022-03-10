import numpy as np


class DataGenerator:
    def input_fun(self, n_samples, dim):
        pass

    def noise_fun(self, n_samples_train):
        pass

    def teacher_fun(self, dim):
        pass

    def y_fun(self, xs, theta):
        pass

    def generate_test_training_set(self, n_samples_train, n_samples_test, dim):
        theta_0_teacher = self.teacher_fun(dim)

        xs = self.input_fun(n_samples_train, dim)
        total_error = self.noise_fun(n_samples_train)

        xs_gen = self.input_fun(n_samples_test, dim)

        ys = self.y_fun(xs, theta_0_teacher) + total_error
        ys_gen = self.y_fun(xs_gen, theta_0_teacher)

        return xs, ys, xs_gen, ys_gen, theta_0_teacher


class DataGenerationSingleNoise(DataGenerator):
    def __init__(self, delta):
        self.delta = delta
        return

    def input_fun(self, n_samples, dim):
        return np.random.normal(loc=0.0, scale=1.0, size=(n_samples, dim))

    def noise_fun(self, n_samples):
        error_sample = np.sqrt(self.delta) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        return error_sample

    def teacher_fun(self, dim):
        return np.random.normal(loc=0.0, scale=1.0, size=(dim,))

    def y_fun(self, xs, theta):
        n, d = xs.shape
        return np.divide(xs @ theta, np.sqrt(d))


class DataGenerationDoubleNoise(DataGenerator):
    def __init__(self, delta_small, delta_large, eps):
        self.delta_small = delta_small
        self.delta_large = delta_large
        self.epsilon = eps
        return

    def input_fun(self, n_samples, dim):
        return np.random.normal(loc=0.0, scale=1.0, size=(n_samples, dim))

    def noise_fun(self, n_samples):
        choice = np.random.choice(
            [0, 1], p=[1 - self.epsilon, self.epsilon], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(self.delta_small) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(self.delta_large) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        return total_error

    def teacher_fun(self, dim):
        return np.random.normal(loc=0.0, scale=1.0, size=(dim,))

    def y_fun(self, xs, theta):
        n, d = xs.shape
        return np.divide(xs @ theta, np.sqrt(d))
