"""CPU regression checks for the GES normalization and weight migration.

Run with: python -m unittest discover -s tests -v
"""
import unittest

import torch

from dvsl.ges import build_arg_parser, smoothness_loss


class GESSmoothnessTests(unittest.TestCase):
    def setUp(self):
        self.displacement = torch.tensor(
            [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]],
            dtype=torch.float64, requires_grad=True,
        )
        self.faces = torch.tensor([[0, 1, 2], [2, 1, 3]])

    def test_coordinate_average_and_repeated_interior_edge(self):
        # Squared lengths: 1, 5, 4 and 5, 10, 13; the shared edge counts twice.
        loss = smoothness_loss(self.displacement, self.faces)
        self.assertAlmostEqual(loss.item(), 38. / 18.)
        gradient, = torch.autograd.grad(loss, self.displacement)
        expected = torch.tensor(
            [[-1., -2., 0.], [4., -4., -3.], [-2., 8., -3.], [-1., -2., 6.]],
            dtype=torch.float64,
        ) / 9.
        torch.testing.assert_close(gradient, expected)

    def test_translation_invariance_and_uniform_displacement(self):
        shift = torch.tensor([7., -3., 2.], dtype=torch.float64)
        torch.testing.assert_close(
            smoothness_loss(self.displacement + shift, self.faces),
            smoothness_loss(self.displacement, self.faces),
        )
        self.assertEqual(smoothness_loss(shift.expand(4, -1), self.faces).item(), 0.)

    def test_migrated_weights_preserve_objective_and_gradient(self):
        edges = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]],
                           self.faces[:, [2, 0]]])
        differences = self.displacement[edges[:, 0]] - self.displacement[edges[:, 1]]
        legacy = differences.square().sum(dim=1).mean()
        normalized = smoothness_loss(self.displacement, self.faces)
        for old_weight in (0.01, 1., 30., 25.):
            with self.subTest(old_weight=old_weight):
                old_loss = old_weight * legacy
                new_loss = (3 * old_weight) * normalized
                torch.testing.assert_close(new_loss, old_loss)
                old_grad, = torch.autograd.grad(old_loss, self.displacement, retain_graph=True)
                new_grad, = torch.autograd.grad(new_loss, self.displacement, retain_graph=True)
                torch.testing.assert_close(new_grad, old_grad)

    def test_cli_default_and_explicit_weights(self):
        parser = build_arg_parser()
        required = ['--pairs', 'source.vtk', 'target.vtk', '--metric', 'pointwise',
                    '--output_dir', 'unused']
        self.assertEqual(parser.parse_args(required).lambda_smooth, 75.)
        for flag in ('--lambda_smooth', '--smoothness_weight'):
            self.assertEqual(parser.parse_args(required + [flag, '0.03']).lambda_smooth, 0.03)


if __name__ == '__main__':
    unittest.main()
