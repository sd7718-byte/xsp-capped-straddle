import pytest

from xsp_straddle.filtering import (
    PreliminarySide,
    conservative_preliminary_screen,
    exact_iv_envelope,
    maybe_screen,
)


def test_twenty_percent_example_matches_brief():
    envelope = exact_iv_envelope(0.20)
    assert envelope.lower == pytest.approx(0.19198)
    assert envelope.upper == pytest.approx(0.22568)


def test_21_percent_forecast_has_no_robust_long_edge_against_20_percent_ask():
    result = conservative_preliminary_screen(0.19, 0.20, 0.21, 0.21, 0.0)
    assert result.side is PreliminarySide.NONE
    assert result.long_margin < 0.0


def test_long_and_short_screens_use_pessimistic_bounds_and_cost():
    long_result = conservative_preliminary_screen(0.17, 0.18, 0.22, 0.24, 0.01)
    assert long_result.side is PreliminarySide.LONG
    short_result = conservative_preliminary_screen(0.25, 0.26, 0.20, 0.22, 0.01)
    assert short_result.side is PreliminarySide.SHORT


def test_optional_screen_can_be_skipped_but_partial_input_is_rejected():
    assert maybe_screen(None, None, 0.2, 0.3, 0.0) is None
    with pytest.raises(ValueError):
        maybe_screen(0.2, None, 0.2, 0.3, 0.0)


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        conservative_preliminary_screen(0.2, 0.19, 0.2, 0.3, 0.0)
    with pytest.raises(ValueError):
        conservative_preliminary_screen(0.19, 0.2, 0.3, 0.2, 0.0)
