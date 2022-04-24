from unittest import TestCase
from src.integration_utils import (
    divide_integration_borders_grid,
    domains_double_line_constraint,
)


def line(x, m, q):
    return m * x + q


class TestDomainSplitters(TestCase):
    def test_domain_double_line_constraint(self):
        borders = [[1.0, -1.0], [1.0, -1.0]]

        cases = [(0.0, 0.5), (0.5, 0.1), (1.0, 0.2), ()]
        out_lens = [3, 4]

        splitted_borders = [
            domains_double_line_constraint(
                borders,
                lambda x: line(x, m, q),
                lambda x: line(x, m, q),
                lambda x: line(x, 0.0, 0.0),
                {},
                {},
                {},
            )
            for (m, q) in cases
        ]

        for b, l in zip(splitted_borders, out_lens):
            self.assertEqual(len(b), l)

    def test_divide_integration_borders_square(self):
        borders = [[1.0, -1.0], [1.0, -1.0]]
        proportions = [0.1, 0.5, 0.9]

        splitted_borders = [
            divide_integration_borders_grid(borders, proportion=p) for p in proportions
        ]

        for b, p in zip(splitted_borders, proportions):
            self.assertEqual(len(b), 9)
            # check that the results are actually good

        bad_proportions = [-0.1, 0.0, 1.0, 1.1]
        for bp in bad_proportions:
            with self.assertRaises(ValueError):
                divide_integration_borders_grid(borders, proportion=bp)


class TestIntegration(TestCase):
    def test_L2_integrations_comparisions_single_noise(self):
        raise NotImplementedError

    def test_BO_integrations_comparisions_single_noise(self):
        raise NotImplementedError
