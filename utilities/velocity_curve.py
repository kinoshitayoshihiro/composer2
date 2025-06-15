PREDEFINED_CURVES = {
    "crescendo": [0.6, 0.8, 1.0],
    "decrescendo": [1.0, 0.8, 0.6],
}


def resolve_velocity_curve(curve_option):
    """Return a list of velocity scale factors for the given option."""
    if curve_option is None:
        return []
    if isinstance(curve_option, str):
        return PREDEFINED_CURVES.get(curve_option.lower(), [])
    if isinstance(curve_option, (list, tuple)):
        try:
            return [float(x) for x in curve_option]
        except Exception:
            return []
    return []
