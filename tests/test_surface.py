import pytest

from xsp_straddle.surface import SurfaceError, SurfacePoint, TotalVarianceSurface


def _surface():
    return TotalVarianceSurface(
        forward=100.0,
        maturity_years=0.25,
        points=[
            SurfacePoint(90.0, 0.24),
            SurfacePoint(100.0, 0.20),
            SurfacePoint(110.0, 0.19),
        ],
    )


def test_zero_shift_reproduces_market_surface():
    surface = _surface()
    for strike in surface.strikes:
        assert surface.shifted_iv(strike, surface.atm_iv) == pytest.approx(
            surface.market_iv(strike)
        )


def test_shift_is_constant_in_total_variance_and_preserves_skew_difference():
    surface = _surface()
    q = 0.25
    expected_shift = 0.25 * (q**2 - 0.20**2)
    shifts = [
        surface.shifted_total_variance(strike, q) - surface.market_total_variance(strike)
        for strike in [90.0, 100.0, 110.0]
    ]
    assert shifts == pytest.approx([expected_shift] * 3)


def test_low_forecast_that_makes_a_wing_variance_nonpositive_is_rejected():
    surface = TotalVarianceSurface(
        100.0,
        1.0,
        [SurfacePoint(80.0, 0.05), SurfacePoint(100.0, 0.30), SurfacePoint(120.0, 0.05)],
    )
    with pytest.raises(SurfaceError, match="non-positive"):
        surface.shifted_iv(80.0, 0.10)


def test_surface_rejects_duplicate_or_nonpositive_inputs():
    with pytest.raises(SurfaceError):
        TotalVarianceSurface(100.0, 0.25, [SurfacePoint(100, 0.2), SurfacePoint(100, 0.3)])
    with pytest.raises(SurfaceError):
        TotalVarianceSurface(100.0, 0.25, [SurfacePoint(90, 0.2), SurfacePoint(110, 0.0)])
