"""Board-geometry conformance for the Nearby Aircraft plugin.

Renders the plugin across every board shape FiestaBoard supports (Flagship,
Note, and note_array panels from 15x3 up to 120x24) using the shared
conformance suite core holds its own plugins to. See
src/plugins/geometry_conformance.py in FiestaBoard for what each check means.

strict_growth=True because "aircraft" is a list: a taller board must show
more of it, not the same fixed handful padded with blank rows.
"""

from unittest.mock import Mock, patch

from src.plugins.geometry_conformance import assert_board_conformance

from plugins.nearby_aircraft import NearbyAircraftPlugin

# Enough distinct, in-radius, airborne aircraft that the tallest board in the
# growth ladder (120x24 -> up to 22 aircraft rows) has real content to grow
# into, rather than the check passing only because there was nothing to show
# either way.
_NUM_MOCK_AIRCRAFT = 25


def _mock_states(base_lat: float, base_lon: float) -> list:
    states = []
    for i in range(_NUM_MOCK_AIRCRAFT):
        states.append(
            [
                f"a{i:05x}",  # icao24
                f"FST{i:03d}",  # callsign
                "US",  # origin_country
                1234567890,  # time_position
                1234567890,  # last_contact
                base_lon + i * 0.001,  # longitude
                base_lat + i * 0.001,  # latitude -- spreads distances so sort is meaningful
                (1000.0 + i * 200),  # baro_altitude (m)
                False,  # on_ground
                120.0 + i,  # velocity (m/s)
                180.0,  # true_track
                0.0,  # vertical_rate
                None,  # sensors
                1000.0 + i * 200,  # geo_altitude (m)
                str(1200 + i).zfill(4),  # squawk
                False,  # spi
                0,  # position_source
            ]
        )
    return states


def _make_manifest() -> dict:
    import json
    from pathlib import Path

    return json.loads((Path(__file__).parent.parent / "manifest.json").read_text())


def test_renders_on_every_board_shape():
    manifest = _make_manifest()
    base_lat, base_lon = 37.7749, -122.4194

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"time": 1234567890, "states": _mock_states(base_lat, base_lon)}

    patcher = patch("plugins.nearby_aircraft.requests.get", return_value=mock_response)
    patcher.start()
    try:
        def make_plugin() -> NearbyAircraftPlugin:
            plugin = NearbyAircraftPlugin(manifest)
            plugin.config = {
                "enabled": True,
                "latitude": base_lat,
                "longitude": base_lon,
                "radius_km": 500,
                # The ceiling (24), not the old Flagship-sized default (4):
                # a plugin capped independently of the board would leave a
                # large panel's rows blank even at max configuration.
                "max_aircraft": 24,
                "refresh_seconds": 300,
            }
            return plugin

        assert_board_conformance(
            make_plugin,
            manifest=manifest,
            strict_growth=True,
            require_note_array_preview=True,
        )
    finally:
        patcher.stop()
