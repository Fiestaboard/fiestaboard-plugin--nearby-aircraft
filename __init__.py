"""Nearby Aircraft plugin for FiestaBoard.

Displays real-time nearby aircraft information using the OpenSky Network API.
Shows call sign, altitude, ground speed, and squawk code for aircraft within
a user-defined radius.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging
import requests
from math import radians, cos, sin, asin, sqrt

from src.devices import NOTE_COLS
from src.plugins.base import PluginBase, PluginResult

logger = logging.getLogger(__name__)

# Absolute largest board FiestaBoard supports: an 8x8 note_array (see
# src/devices.py MAX_NOTES_PER_AXIS). max_aircraft's settings_schema maximum
# is sized to this so a large FiestaPanel isn't starved by a cap meant for a
# Flagship -- the actual number shown on any given board is still clamped to
# what that board's rows can hold (see get_formatted_display).
MAX_NOTES_PER_AXIS = 8
NOTE_ROWS = 3
MAX_POSSIBLE_ROWS = MAX_NOTES_PER_AXIS * NOTE_ROWS
MAX_AIRCRAFT_CEILING = MAX_POSSIBLE_ROWS

# Width budget for the single-line 'formatted'/'headers' template variables.
# This is a content contract (manifest max_lengths declares both at 22), not
# a board dimension -- it never grows past 22 even on a wide note_array (the
# page editor already trusts a variable declared at 22 to fit anywhere a user
# places it), and only ever shrinks for a Note-width board.
DECLARED_LINE_WIDTH = 22

# OpenSky API endpoints
OPENSKY_BASE_URL = "https://opensky-network.org/api"
OPENSKY_OAUTH_URL = f"{OPENSKY_BASE_URL}/v1/oauth/token"
OPENSKY_STATES_URL = f"{OPENSKY_BASE_URL}/states/all"

# State vector field indices (OpenSky API returns arrays)
# See: https://openskynetwork.github.io/opensky-api/rest.html#response
STATE_VECTOR_INDICES = {
    "icao24": 0,
    "callsign": 1,
    "origin_country": 2,
    "time_position": 3,
    "last_contact": 4,
    "longitude": 5,
    "latitude": 6,
    "baro_altitude": 7,
    "on_ground": 8,
    "velocity": 9,
    "true_track": 10,
    "vertical_rate": 11,
    "sensors": 12,
    "geo_altitude": 13,
    "squawk": 14,
    "spi": 15,
    "position_source": 16,
}


class NearbyAircraftPlugin(PluginBase):
    """Nearby aircraft plugin.
    
    Fetches real-time aircraft data from OpenSky Network API and displays
    aircraft within a specified radius showing call sign, altitude, ground
    speed, and squawk code.
    """
    
    def __init__(self, manifest: Dict[str, Any]):
        """Initialize the nearby aircraft plugin."""
        super().__init__(manifest)
        self._cache: Optional[Dict[str, Any]] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
    
    @property
    def plugin_id(self) -> str:
        return "nearby_aircraft"
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate nearby aircraft configuration."""
        errors = []
        
        lat = config.get("latitude")
        lon = config.get("longitude")
        
        if lat is None:
            errors.append("Latitude is required")
        elif not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
            errors.append("Latitude must be a number between -90 and 90")
        
        if lon is None:
            errors.append("Longitude is required")
        elif not isinstance(lon, (int, float)) or not (-180 <= lon <= 180):
            errors.append("Longitude must be a number between -180 and 180")
        
        radius_km = config.get("radius_km", 50)
        if not isinstance(radius_km, (int, float)) or radius_km < 1:
            errors.append("Radius must be at least 1 km")
        
        max_aircraft = config.get("max_aircraft", 4)
        if not isinstance(max_aircraft, int) or not (1 <= max_aircraft <= MAX_AIRCRAFT_CEILING):
            errors.append(f"Max aircraft must be between 1 and {MAX_AIRCRAFT_CEILING}")
        
        refresh_seconds = config.get("refresh_seconds", 120)
        if not isinstance(refresh_seconds, int) or refresh_seconds < 10:
            errors.append("Refresh interval must be at least 10 seconds")
        
        return errors
    
    def on_config_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Drop cached aircraft and token so a config change takes effect immediately.
        
        The cache is keyed only on age, so without this a change to
        `latitude`/`longitude`/`radius_km` would keep serving aircraft found
        around the old position for up to refresh_seconds. The access token
        is minted from `client_id`/`client_secret`, so it goes too.
        """
        self._cache = None
        self._access_token = None
        self._token_expires_at = None
        logger.debug("Cleared cached aircraft after config change")
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth in km.
        
        Uses the Haversine formula to compute great-circle distance.
        
        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in kilometers
        return 6371 * c
    
    @staticmethod
    def calculate_bounding_box(lat: float, lon: float, radius_km: float) -> Dict[str, float]:
        """Calculate bounding box from center point and radius.
        
        Args:
            lat: Center latitude in degrees
            lon: Center longitude in degrees
            radius_km: Radius in kilometers
            
        Returns:
            Dictionary with lamin, lamax, lomin, lomax
        """
        # Approximate conversion: 1 degree latitude ≈ 111 km
        # For longitude, adjust by latitude (cos(lat))
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * abs(cos(radians(lat))))
        
        return {
            "lamin": lat - lat_delta,
            "lamax": lat + lat_delta,
            "lomin": lon - lon_delta,
            "lomax": lon + lon_delta,
        }
    
    def _get_access_token(self) -> Optional[str]:
        """Get OAuth2 access token for authenticated requests.
        
        Returns:
            Access token string, or None if authentication fails
        """
        client_id = self.config.get("client_id", "").strip()
        client_secret = self.config.get("client_secret", "").strip()
        
        # If no credentials, return None (use unauthenticated)
        if not client_id or not client_secret:
            return None
        
        # Check if we have a valid cached token
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at:
                return self._access_token
        
        # Request new token
        try:
            response = requests.post(
                OPENSKY_OAUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to get OAuth token: {response.status_code}")
                return None
            
            data = response.json()
            self._access_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)  # Default 1 hour
            
            # Set expiration time (subtract 60 seconds for safety margin)
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
            
            logger.debug("Successfully obtained OAuth access token")
            return self._access_token
            
        except Exception as e:
            logger.error(f"Error getting OAuth token: {e}")
            return None
    
    def _get_api_headers(self) -> Dict[str, str]:
        """Get headers for API requests, including auth if available."""
        headers = {}
        token = self._get_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
    
    def _parse_state_vector(self, state_vector: List[Any]) -> Optional[Dict[str, Any]]:
        """Parse a single state vector from OpenSky API.
        
        Args:
            state_vector: Array of state vector values from API
            
        Returns:
            Parsed aircraft data dictionary, or None if invalid
        """
        try:
            if not state_vector or len(state_vector) < 17:
                return None
            
            # Extract fields using indices
            icao24 = state_vector[STATE_VECTOR_INDICES["icao24"]]
            callsign_raw = state_vector[STATE_VECTOR_INDICES["callsign"]]
            latitude = state_vector[STATE_VECTOR_INDICES["latitude"]]
            longitude = state_vector[STATE_VECTOR_INDICES["longitude"]]
            baro_altitude = state_vector[STATE_VECTOR_INDICES["baro_altitude"]]
            geo_altitude = state_vector[STATE_VECTOR_INDICES["geo_altitude"]]
            velocity = state_vector[STATE_VECTOR_INDICES["velocity"]]
            squawk_raw = state_vector[STATE_VECTOR_INDICES["squawk"]]
            on_ground = state_vector[STATE_VECTOR_INDICES["on_ground"]]
            
            # Skip aircraft on ground
            if on_ground:
                return None
            
            # Skip if no position data
            if latitude is None or longitude is None:
                return None
            
            # Get callsign - use ICAO address if missing
            if callsign_raw:
                callsign = str(callsign_raw).strip()
            else:
                # Use ICAO 24-bit address as fallback
                callsign = str(icao24) if icao24 else "UNKNOWN"
            
            # Get altitude - prefer geo_altitude, fallback to baro_altitude
            altitude_m = None
            if geo_altitude is not None:
                altitude_m = float(geo_altitude)
            elif baro_altitude is not None:
                altitude_m = float(baro_altitude)
            
            if altitude_m is None:
                return None  # Skip aircraft without altitude
            
            # Convert altitude from meters to feet
            altitude_ft = int(altitude_m * 3.28084)
            
            # Get velocity (ground speed) in m/s, convert to knots
            if velocity is None:
                return None  # Skip aircraft without velocity
            
            velocity_ms = float(velocity)
            ground_speed_knots = int(velocity_ms * 1.94384)
            
            # Get squawk code
            # Squawk can be None, empty string, 0, or a valid 4-digit code
            if squawk_raw is not None and squawk_raw != "":
                try:
                    squawk_int = int(float(squawk_raw))  # Handle string numbers
                    if squawk_int == 0:
                        # 0 means no squawk code assigned
                        squawk = "----"
                    else:
                        squawk = str(squawk_int).zfill(4)
                except (ValueError, TypeError):
                    squawk = "----"
            else:
                squawk = "----"
            
            # Note: no "formatted" field here. The board a plugin renders on
            # is not known until get_data(board) binds self.board, and
            # get_data()'s cache (and this plugin's own self._cache, see
            # on_config_change) is geometry-agnostic, so a per-board string
            # baked in here would leak one board's column widths into
            # another's render. _align_formatting derives "formatted" fresh
            # from these raw fields every time, bound to self.board.
            return {
                "icao24": str(icao24) if icao24 else "",
                "call_sign": callsign,
                "altitude": altitude_ft,
                "ground_speed": ground_speed_knots,
                "squawk": squawk,
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
            
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Error parsing state vector: {e}")
            return None
    
    def _render_width(self) -> int:
        """Column budget for the single-line 'formatted'/'headers' variables.

        Derived from ``self.board`` (defaulting to a Flagship's 22 when
        unbound, matching the platform contract for a legacy/unscoped call),
        then capped at :data:`DECLARED_LINE_WIDTH`. A note_array's ``cols``
        is always a multiple of :data:`NOTE_COLS` (15) and a Flagship's is
        22, so this cap can only ever land on exactly 15 or 22 -- there is no
        board whose width falls strictly between them -- which is why the
        two formatting branches below only need to handle those two cases.
        """
        board = self.board
        cols = board.cols if board is not None else DECLARED_LINE_WIDTH
        return min(cols, DECLARED_LINE_WIDTH)

    def _align_formatting(self, aircraft_list: List[Dict]) -> Tuple[List[Dict], str]:
        """Derive 'formatted'/'headers' for *aircraft_list*, sized to the board.

        Recomputed from each aircraft's raw fields every call rather than
        read from a previously-baked string: the raw aircraft dicts (and
        this plugin's own network-avoidance cache in ``self._cache``) carry
        no board-specific formatting, so the same cached data renders
        correctly whichever board is currently bound to ``self.board``. A
        Note-width board (15 cols) drops the squawk column and abbreviates
        headers rather than truncating every field to fit a layout designed
        for 22; every wider board (including every note_array at least two
        notes wide) uses the classic layout.

        Args:
            aircraft_list: List of aircraft dictionaries (mutated in place;
                each gets a "formatted" key).

        Returns:
            Tuple of (aircraft list, headers string).
        """
        width = self._render_width()
        if width <= NOTE_COLS:
            return self._align_formatting_narrow(aircraft_list, width)
        return self._align_formatting_wide(aircraft_list, width)

    def _align_formatting_wide(self, aircraft_list: List[Dict], width: int) -> Tuple[List[Dict], str]:
        """Classic 4-column layout (call sign, altitude, speed, squawk)."""
        if not aircraft_list:
            return aircraft_list, "CALLSGN ALT GS SQWK"

        # Calculate max widths for each field
        max_call_sign_width = 0
        max_altitude_width = 0
        max_speed_width = 0

        for aircraft in aircraft_list:
            call_sign = str(aircraft.get("call_sign", ""))
            altitude = str(aircraft.get("altitude", 0))
            ground_speed = str(aircraft.get("ground_speed", 0))

            max_call_sign_width = max(max_call_sign_width, len(call_sign))
            max_altitude_width = max(max_altitude_width, len(altitude))
            max_speed_width = max(max_speed_width, len(ground_speed))

        # Cap call sign/altitude/speed so the combined line can never grow
        # past `width` regardless of how long any one field's content is.
        max_call_sign_width = min(max_call_sign_width, 8)
        max_altitude_width = min(max(max_altitude_width, 5), 6)  # 5-6 digits
        max_speed_width = min(max(max_speed_width, 3), 4)  # 3-4 digits
        max_squawk_width = 4  # Squawk is always 4

        # Generate aligned headers. ljust/rjust never truncate, so a label
        # longer than its column (e.g. "CALLSGN" in a narrower column) still
        # renders whole; the final slice below is the actual width guard.
        callsign_header = "CALLSGN".ljust(max_call_sign_width)
        altitude_header = "ALT".rjust(max_altitude_width)
        speed_header = "GS".rjust(max_speed_width)
        squawk_header = "SQWK".rjust(max_squawk_width)
        headers = f"{callsign_header} {altitude_header} {speed_header} {squawk_header}"
        headers = headers[:width]

        # Apply aligned formatting to all aircraft
        for aircraft in aircraft_list:
            call_sign = str(aircraft.get("call_sign", ""))[:max_call_sign_width]
            altitude = aircraft.get("altitude", 0)
            ground_speed = aircraft.get("ground_speed", 0)
            squawk = str(aircraft.get("squawk", "----"))[:4]

            call_sign_display = call_sign.ljust(max_call_sign_width)
            altitude_str = str(altitude).rjust(max_altitude_width)
            speed_str = str(ground_speed).rjust(max_speed_width)
            squawk_str = squawk.rjust(max_squawk_width)

            # Combine: "CALLSGN  ALT   GS  SQWK"
            formatted = f"{call_sign_display} {altitude_str} {speed_str} {squawk_str}"
            aircraft["formatted"] = formatted[:width]

        return aircraft_list, headers

    @staticmethod
    def _align_formatting_narrow(aircraft_list: List[Dict], width: int) -> Tuple[List[Dict], str]:
        """Abbreviated 3-column layout for a Note-width board (15 cols).

        The classic layout needs 25 characters at minimum; a Note only has
        15. Squawk is dropped first (least useful for a glance at what's
        overhead) and labels are abbreviated so headers still fit.
        """
        call_w, alt_w, speed_w = 5, 5, 3  # 5 + 1 + 5 + 1 + 3 == 15 == NOTE_COLS

        headers = f"{'CALL'.ljust(call_w)} {'ALT'.rjust(alt_w)} {'GS'.rjust(speed_w)}"[:width]
        if not aircraft_list:
            return aircraft_list, headers

        for aircraft in aircraft_list:
            call_sign = str(aircraft.get("call_sign", ""))[:call_w].ljust(call_w)
            altitude = str(aircraft.get("altitude", 0))[:alt_w].rjust(alt_w)
            ground_speed = str(aircraft.get("ground_speed", 0))[:speed_w].rjust(speed_w)
            formatted = f"{call_sign} {altitude} {ground_speed}"
            aircraft["formatted"] = formatted[:width]

        return aircraft_list, headers

    def _format_aircraft_line(self, call_sign: str, altitude: int, ground_speed: int, squawk: str) -> str:
        """Format one aircraft's data for display, sized to the current board.

        Standalone formatter (no alignment across a list) kept for direct
        use and for the tests that exercise it. Bound to ``self.board`` the
        same way :meth:`_align_formatting` is, via :meth:`_render_width`, so
        it never emits a row wider than the board actually has.

        Args:
            call_sign: Aircraft call sign
            altitude: Altitude in feet
            ground_speed: Ground speed in knots
            squawk: Squawk code (4 chars)

        Returns:
            Formatted string, at most ``self._render_width()`` chars.
        """
        width = self._render_width()
        if width <= NOTE_COLS:
            call_sign_display = call_sign[:5].ljust(5)
            altitude_str = str(altitude)[:5].rjust(5)
            speed_str = str(ground_speed)[:3].rjust(3)
            formatted = f"{call_sign_display} {altitude_str} {speed_str}"
            return formatted[:width]

        # Truncate and pad call sign to 8 chars (left-aligned)
        call_sign_display = call_sign[:8].ljust(8)

        # Format altitude (right-aligned, 5-6 digits)
        altitude_str = str(altitude).rjust(6)

        # Format ground speed (right-aligned, 3-4 digits)
        speed_str = str(ground_speed).rjust(4)

        # Format squawk (right-aligned, 4 chars)
        squawk_str = squawk[:4].rjust(4)

        # Combine: "CALLSGN  ALT   GS  SQWK"
        formatted = f"{call_sign_display} {altitude_str} {speed_str} {squawk_str}"
        return formatted[:width]
    
    def _empty_result_data(self) -> Dict[str, Any]:
        """Payload for 'no aircraft nearby', sized to the current board.

        Built the same board-aware way as a populated result (via
        :meth:`_align_formatting`) so a Note-width board never gets handed
        the 19-character literal "NO AIRCRAFT NEARBY", which alone is wider
        than a Note (15 cols).
        """
        _, headers = self._align_formatting([])
        width = self._render_width()
        message = "NO AIRCRAFT NEARBY" if width >= len("NO AIRCRAFT NEARBY") else "NO AIRCRAFT"
        return {
            "aircraft_count": 0,
            "aircraft": [],
            "call_sign": "",
            "altitude": 0,
            "ground_speed": 0,
            "squawk": "----",
            "formatted": message[:width],
            "headers": headers,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _build_result_data(self, raw_aircraft: List[Dict], last_updated: str) -> Dict[str, Any]:
        """Build the data dict for *raw_aircraft*, formatting fresh for the current board.

        ``raw_aircraft`` holds only geometry-independent fields (no
        "formatted"/"headers" baked in -- see :meth:`_parse_state_vector`),
        whether it came straight from the network or from ``self._cache``.
        Copying before calling :meth:`_align_formatting` (which mutates its
        argument) keeps the cached raw dicts themselves geometry-agnostic
        across repeated calls, which is what lets the very same cached fetch
        render correctly on whichever board is bound to ``self.board`` this
        time (see the F6 note in on_config_change and the module docstring
        on _parse_state_vector).
        """
        if not raw_aircraft:
            return self._empty_result_data()

        aircraft = [dict(a) for a in raw_aircraft]
        aircraft, headers = self._align_formatting(aircraft)
        primary = aircraft[0]

        return {
            "call_sign": primary["call_sign"],
            "altitude": primary["altitude"],
            "ground_speed": primary["ground_speed"],
            "squawk": primary["squawk"],
            "formatted": primary["formatted"],
            "headers": headers,
            "aircraft_count": len(aircraft),
            "last_updated": last_updated,
            "aircraft": aircraft,
        }

    def _max_aircraft_config(self) -> int:
        """The user's configured aircraft cap, clamped to sane bounds.

        A ceiling independent of any one board (see MAX_AIRCRAFT_CEILING) --
        how many of those are actually *shown* is decided per render in
        :meth:`_build_display_lines`, from ``self.board.rows``.
        """
        max_aircraft = self.config.get("max_aircraft", 4)
        if not isinstance(max_aircraft, int) or max_aircraft < 1:
            max_aircraft = 4
        return min(max_aircraft, MAX_AIRCRAFT_CEILING)

    def _build_display_lines(self, data: Dict[str, Any]) -> List[str]:
        """This plugin's whole-board display for *data*, sized to ``self.board``.

        Shared by :meth:`fetch_data` (``PluginResult.formatted_lines`` -- the
        path a page actually renders when it displays this plugin directly,
        see ``src/displays/service.py``) and :meth:`get_formatted_display`
        (the documented hook core does not call today), so both react to the
        bound board identically and can't drift apart.

        Title and headers are decoration around the actual data, so each is
        only included while at least one row remains for an aircraft
        afterward -- a Note (3 rows) keeps both and shows one aircraft;
        nothing in FiestaBoard's board catalog is short enough to need to
        drop them, but this degrades instead of overflowing if that changes.
        The number of aircraft rows is the smallest of: the user's
        max_aircraft config, the rows actually left on this board, and how
        many aircraft there are -- never a literal count baked in here.
        """
        board = self.board
        rows = board.rows if board is not None else 6
        cols = board.cols if board is not None else 22

        aircraft = data.get("aircraft") or []
        max_aircraft = self._max_aircraft_config()

        lines: List[str] = []
        remaining_rows = rows
        if remaining_rows > 2:
            lines.append("NEARBY AIRCRAFT".center(cols)[:cols])
            remaining_rows -= 1
        headers = str(data.get("headers") or "")
        if headers and remaining_rows > 1:
            lines.append(headers[:cols])
            remaining_rows -= 1

        shown = max(0, min(max_aircraft, remaining_rows, len(aircraft)))
        for ac in aircraft[:shown]:
            lines.append(str(ac.get("formatted", ""))[:cols])

        while len(lines) < rows:
            lines.append("")
        return lines[:rows]

    def _success_result(self, raw_aircraft: List[Dict], last_updated: str) -> PluginResult:
        """Build the full PluginResult (data + formatted_lines) for *raw_aircraft*."""
        data = self._build_result_data(raw_aircraft, last_updated)
        return PluginResult(available=True, data=data, formatted_lines=self._build_display_lines(data))

    def _empty_success_result(self) -> PluginResult:
        """PluginResult for 'no aircraft nearby', sized to the current board."""
        data = self._empty_result_data()
        return PluginResult(available=True, data=data, formatted_lines=self._build_display_lines(data))

    def fetch_data(self) -> PluginResult:
        """Fetch nearby aircraft data from OpenSky API."""
        lat = self.config.get("latitude")
        lon = self.config.get("longitude")
        radius_km = self.config.get("radius_km", 50)
        max_aircraft = self._max_aircraft_config()
        refresh_seconds = self.config.get("refresh_seconds", 120)

        if lat is None or lon is None:
            return PluginResult(
                available=False,
                error="Latitude and longitude are required"
            )

        # Check cache first. self._cache holds only raw, geometry-independent
        # aircraft fields (see _build_result_data), so reusing it across a
        # config-unchanged, refresh-window-fresh call is safe even when
        # self.board differs from the board that was bound when it was
        # populated -- formatting is derived fresh below either way.
        if self._cache and self._cache.get("aircraft"):
            last_updated = self._cache.get("last_updated", "")
            if last_updated:
                try:
                    cache_time = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                    age_seconds = (datetime.now(cache_time.tzinfo) - cache_time).total_seconds()
                    if age_seconds < refresh_seconds:
                        logger.debug(f"Using cached data (age: {age_seconds:.0f}s < {refresh_seconds}s)")
                        return self._success_result(self._cache["aircraft"], self._cache["last_updated"])
                except Exception:
                    pass  # If cache time parsing fails, continue to fetch

        try:
            # Calculate bounding box
            bbox = self.calculate_bounding_box(lat, lon, radius_km)
            
            # Fetch state vectors from OpenSky API
            headers = self._get_api_headers()
            params = {
                "lamin": bbox["lamin"],
                "lamax": bbox["lamax"],
                "lomin": bbox["lomin"],
                "lomax": bbox["lomax"],
            }
            
            response = requests.get(OPENSKY_STATES_URL, params=params, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("OpenSky API rate limit exceeded, using cached data if available")
                if self._cache and self._cache.get("aircraft"):
                    return self._success_result(self._cache["aircraft"], self._cache["last_updated"])
                return PluginResult(
                    available=False,
                    error="API rate limit exceeded. Please wait or use authentication."
                )

            if response.status_code != 200:
                logger.error(f"OpenSky API error: {response.status_code}")
                if self._cache and self._cache.get("aircraft"):
                    return self._success_result(self._cache["aircraft"], self._cache["last_updated"])
                return PluginResult(
                    available=False,
                    error=f"API error: {response.status_code}"
                )

            data = response.json()
            states = data.get("states")

            if not states:
                # No aircraft found, return empty result
                return self._empty_success_result()

            # Parse state vectors and filter by distance
            nearby_aircraft = []
            for state_vector in states:
                aircraft = self._parse_state_vector(state_vector)
                if aircraft:
                    # Calculate actual distance
                    distance_km = self.haversine_distance(
                        lat, lon,
                        aircraft["latitude"], aircraft["longitude"]
                    )

                    if distance_km <= radius_km:
                        aircraft["distance_km"] = round(distance_km, 1)
                        nearby_aircraft.append(aircraft)

            # Sort by distance and limit to the user's configured max_aircraft.
            # This bounds the *fetched/cached* pool, independent of any one
            # board's row count -- how many of these are actually shown is
            # decided per render in get_formatted_display, from board.rows.
            nearby_aircraft.sort(key=lambda a: a.get("distance_km", float('inf')))
            nearby_aircraft = nearby_aircraft[:max_aircraft]

            if not nearby_aircraft:
                return self._empty_success_result()

            last_updated = datetime.now(timezone.utc).isoformat()
            # Cache raw aircraft only -- no "formatted"/"headers" baked in,
            # so this same cached fetch renders correctly for whichever
            # board is bound on a later call (see _build_result_data).
            self._cache = {"aircraft": nearby_aircraft, "last_updated": last_updated}
            return self._success_result(nearby_aircraft, last_updated)

        except requests.exceptions.RequestException as e:
            logger.exception("Error fetching aircraft data")
            if self._cache and self._cache.get("aircraft"):
                return self._success_result(self._cache["aircraft"], self._cache["last_updated"])
            return PluginResult(available=False, error=f"Network error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error fetching aircraft data")
            if self._cache and self._cache.get("aircraft"):
                return self._success_result(self._cache["aircraft"], self._cache["last_updated"])
            return PluginResult(available=False, error=str(e))

    def get_formatted_display(self) -> Optional[List[str]]:
        """Return the plugin's own whole-board display, sized to the board.

        Documented plugin contract (no caller in core today, but conformance
        holds it to the same bounds as the live template-variable path --
        the exact same lines fetch_data now sets as PluginResult.formatted_lines,
        via the shared :meth:`_build_display_lines`). Always calls
        :meth:`fetch_data` -- which serves from its own age-based raw cache
        when fresh, so this rarely touches the network -- rather than
        reading ``self._cache`` directly, because only fetch_data derives
        "formatted"/"headers" for the board that is *currently* bound to
        ``self.board``. Reading a previously baked display straight out of
        the cache would replay whichever board's column widths and row
        count happened to be active on the call that populated it.
        """
        result = self.fetch_data()
        if not result.available or not result.data:
            return None
        return result.formatted_lines or self._build_display_lines(result.data)


# Export the plugin class
Plugin = NearbyAircraftPlugin
