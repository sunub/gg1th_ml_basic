TARGET = "energy_consumption_kwhper100km"

BASIC_FEATURES = [
    "speed_kmh",
    "ambient_temp_C",
    "hvac_power_kw",
    "driving_style_index",
    "tire_pressure_bar",
    "trip_distance_km",
]

EXPANDED_FEATURES = [
    *BASIC_FEATURES,
    "payload_kg",
    "battery_temp_C",
]
