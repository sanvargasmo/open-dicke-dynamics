import numpy as np
import pytest

from open_dicke_mean_field import (
    _rhs_mf_punto,
    analizar_punto_campo_medio,
    puntos_y_modos_campo_medio,
    barrer_lambda_campo_medio,
)


@pytest.mark.parametrize("r", [0.25, 0.5])
def test_fixed_points_and_spin_length(r):
    result = puntos_y_modos_campo_medio(0.9, longitud_espin=r)
    for point in result["puntos"].values():
        assert point["residuo"] < 1e-12
        assert np.linalg.norm(point["estado"][2:]) == pytest.approx(r)


def test_jacobian_and_conserved_spin_length():
    q = np.array([0.2, -0.1, 0.1, 0.2, -0.4])
    args = (1.2, 0.8, 0.6, 0.1)
    point = analizar_punto_campo_medio(q, *args)
    h = 1e-6
    numeric = np.column_stack([
        (_rhs_mf_punto(q+h*e, *args)-_rhs_mf_punto(q-h*e, *args))/(2*h)
        for e in np.eye(5)
    ])
    np.testing.assert_allclose(point["jacobiano"], numeric, atol=1e-10)
    assert np.dot(q[2:], _rhs_mf_punto(q, *args)[2:]) == pytest.approx(0, abs=1e-14)


def test_normal_characteristic_polynomial():
    w, w0, lam, kappa, r = 1.2, 0.8, 0.2, 0.15, 0.3
    poles = puntos_y_modos_campo_medio(
        lam, omega=w, omega_0=w0, kappa=kappa, longitud_espin=r
    )["seleccionado"]["autovalores"]
    polynomial = ((poles+kappa/2)**2+w*w)*(poles*poles+w0*w0)-8*r*lam*lam*w*w0
    np.testing.assert_allclose(polynomial, 0, atol=1e-12)


def test_closed_normal_mode_limit():
    point = puntos_y_modos_campo_medio(0.3, kappa=0)["seleccionado"]
    frequencies = [point["ramas"][name]["omega"] for name in ("LP", "UP")]
    np.testing.assert_allclose(frequencies, np.sqrt([0.4, 1.6]), atol=1e-12)
    np.testing.assert_allclose(point["autovalores"].real, 0, atol=1e-12)


def test_superradiant_symmetry():
    points = puntos_y_modos_campo_medio(0.6)["puntos"]
    np.testing.assert_allclose(
        np.sort_complex(points["superradiante_+1"]["autovalores"]),
        np.sort_complex(points["superradiante_-1"]["autovalores"]), atol=1e-12
    )


def test_degenerate_frequency_preserves_distinct_rates():
    curves = barrer_lambda_campo_medio([0.01], kappa=0.1)
    assert curves["degenerada"][0]
    assert np.isnan(curves["gamma_LP"][0])
    assert np.isnan(curves["gamma_UP"][0])
    rates = curves["gamma_degeneradas"][0]
    assert np.all(np.isfinite(rates))
    assert abs(rates[0]-rates[1]) > 0.01


def test_nonoscillatory_poles_remain_available():
    result = puntos_y_modos_campo_medio(0.6, fase="normal")
    point = result["seleccionado"]
    real_poles = point["autovalores"][abs(point["autovalores"].imag) < 1e-10]
    assert len(real_poles) == 2
    mode = next(m for m in point["modos"] if not m["oscilatorio"])
    assert np.isnan(mode["gamma"])
    np.testing.assert_allclose(mode["gammas"], np.sort(-real_poles.real))
