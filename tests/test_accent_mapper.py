from generator.drum_generator import AccentMapper


def test_accent_mapper_velocity_and_ghost_density():
    mapper = AccentMapper(threshold=0.6, ghost_density_range=(0.3, 0.8))
    base_velocity = 80
    vel_high = mapper.get_velocity(0.8, base_velocity)
    vel_low = mapper.get_velocity(0.2, base_velocity)
    assert vel_high > base_velocity
    assert vel_high >= vel_low
    dens_high = mapper.ghost_density(0.8)
    dens_low = mapper.ghost_density(0.2)
    assert dens_high < dens_low
