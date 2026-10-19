import unittest

import numpy as np

from bandori.raised_objective_experiment import raise_objective_share


class RaisedObjectiveTests(unittest.TestCase):
    def test_objective_group_reaches_share_and_preserves_within_group_ratios(self):
        rng=np.random.default_rng(3)
        weights=rng.random((11,2,6));tiers=[100,500,1000,2000,5000,20000]
        raised,info=raise_objective_share(weights,tiers)
        columns=np.array([t in (500,1000,2000) for t in tiers])
        self.assertAlmostEqual(float(raised[...,columns].sum()/raised.sum()),.6,places=12)
        self.assertAlmostEqual(info['objective_mass_before'],float(weights[...,columns].sum()/weights.sum()),places=12)
        for group in (columns,~columns):
            before=weights[...,group].reshape(-1)/weights[...,group].sum()
            after=raised[...,group].reshape(-1)/raised[...,group].sum()
            np.testing.assert_allclose(after,before,rtol=1e-12)

    def test_zero_weights_stay_zero_and_four_dimensional_input_works(self):
        weights=np.ones((5,2,3,4));weights[...,2]=0;tiers=[500,1000,5000,20000]
        raised,info=raise_objective_share(weights,tiers,share=.6)
        self.assertTrue(np.all(raised[...,2]==0))
        self.assertAlmostEqual(float(raised[...,:2].sum()/raised.sum()),.6,places=12)
        self.assertEqual(raised.shape,weights.shape)
        self.assertEqual(raised.dtype,weights.dtype)

    def test_rejects_degenerate_mass(self):
        with self.assertRaises(ValueError):
            raise_objective_share(np.zeros((3,2,2)),[500,1000])
        with self.assertRaises(ValueError):
            raise_objective_share(np.ones((3,2,2)),[500,1000])  # no non-objective tier mass
        with self.assertRaises(ValueError):
            raise_objective_share(np.ones((3,2,2)),[7,9])


if __name__=='__main__':
    unittest.main()
