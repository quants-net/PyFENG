import numpy as np
import pytest
from scipy.integrate import quad

from pyfeng.asian import BsmAsianJsu


@pytest.mark.parametrize("carry", [0.0, 1e-12, -1e-12, 0.04, -0.03])
@pytest.mark.parametrize("texp", [0.25, 1.0, 3.0])
def test_asian_first_moment_matches_expected_forward_average(carry, texp):
    model = BsmAsianJsu(sigma=0.2, intr=0.02 + carry, divr=0.02)
    expected = quad(lambda t: np.exp(carry * t), 0.0, texp)[0] / texp

    np.testing.assert_allclose(model.price_mnc(texp), expected, rtol=1e-13)


def test_asian_first_moment_broadcasts_and_does_not_depend_on_volatility():
    expiry = np.array([0.25, 1.0, 3.0])
    model = BsmAsianJsu(sigma=np.array([0.0, 0.2, 0.5]), intr=0.02, divr=0.02)

    np.testing.assert_array_equal(model.price_mnc(expiry), np.ones(3))
