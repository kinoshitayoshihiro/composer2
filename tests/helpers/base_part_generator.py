import pytest
from unittest.mock import Mock, patch, call
from music21 import stream, note, instrument

from generator.base_part_generator import BasePartGenerator  # , OverrideModelType (type hint)

# Path for mocks targeting the module where the tested function resides
BGP_PATH = "generator.base_part_generator"

# A mock class for Pydantic-like models (e.g., PartOverride)
class MockPydanticModel:
    def __init__(self, **kwargs):
        self._data = kwargs
        # Set attributes for direct access, e.g., self.overrides.swing_ratio
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self, exclude_unset=True):
        # Simplified model_dump for testing logging
        # Pydantic's exclude_unset=True only includes fields that were explicitly set.
        # For this mock, we'll consider non-None values as "set".
        return {k: v for k, v in self._data.items() if v is not None} if exclude_unset else self._data.copy()

    def __getattr__(self, name):
        # Fallback for attributes not explicitly set during __init__
        # This helps with getattr(self.overrides, "some_attr", None) patterns
        try:
            return self._data[name]
        except KeyError:
            # Mimic Pydantic behavior where unset fields might not be attributes
            # or return None if that's the desired default for getattr.
            # For this mock, raising AttributeError is closer to Pydantic if a field truly doesn't exist.
            # However, getattr with a default handles this, so None is fine for testing.
            return None


class ConcreteTestGenerator(BasePartGenerator):
    """A concrete implementation of BasePartGenerator for testing."""
    def __init__(self, part_name="test_part", **kwargs):
        super().__init__(
            part_name=part_name,
            default_instrument=instrument.Piano(), # A default instrument
            global_tempo=kwargs.pop("global_tempo", 120),
            global_time_signature=kwargs.pop("global_time_signature", "4/4"),
            global_key_signature_tonic=kwargs.pop("global_key_signature_tonic", "C"),
            global_key_signature_mode=kwargs.pop("global_key_signature_mode", "major"),
            **kwargs
        )
        # This mock method will be configured by each test
        self._render_part_mock_method = Mock(return_value=stream.Part(id=f"{part_name}_rendered"))

    def _render_part(
        self,
        section_data: dict,
        next_section_data: dict | None = None,
    ) -> stream.Part | dict[str, stream.Part]:
        return self._render_part_mock_method(section_data, next_section_data)

@pytest.fixture
def mock_logger():
    """Fixture to mock the logger used by BasePartGenerator instances."""
    with patch(f"{BGP_PATH}.logging.getLogger") as mock_get_logger:
        mock_log_instance = Mock()
        mock_get_logger.return_value = mock_log_instance
        yield mock_log_instance

@pytest.fixture
def test_generator(mock_logger): # mock_logger ensures logging is patched for the generator
    """Provides a ConcreteTestGenerator instance for tests."""
    return ConcreteTestGenerator()

@pytest.fixture
def default_section_data():
    """Provides a default section_data dictionary for tests."""
    return {
        "section_name": "TestSection",
        "q_length": 4.0,
        "musical_intent": {"intensity": "medium"},
        "part_params": {},
        # Ensure shared_tracks is initialized, as compose modifies it.
        "shared_tracks": {}
    }

@patch(f"{BGP_PATH}.apply_envelope")
@patch(f"{BGP_PATH}.apply_swing")
@patch(f"{BGP_PATH}.humanize_apply")
@patch(f"{BGP_PATH}.apply_humanization_to_part")
@patch(f"{BGP_PATH}.apply_groove_pretty")
@patch(f"{BGP_PATH}.load_groove_profile")
@patch(f"{BGP_PATH}.get_part_override")
def test_compose_single_part_basic(
    mock_get_part_override: Mock,
    mock_load_groove_profile: Mock,
    mock_apply_groove_pretty: Mock,
    mock_apply_humanization: Mock,
    mock_humanize_apply_base: Mock,
    mock_apply_swing: Mock,
    mock_apply_envelope: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    rendered_part = stream.Part(id="single_rendered")
    rendered_part.append(note.Note("C4")) # Part has notes
    test_generator._render_part_mock_method.return_value = rendered_part
    mock_get_part_override.return_value = None # No overrides

    section_data_input = default_section_data.copy()

    # Act
    result_part = test_generator.compose(section_data=section_data_input)

    # Assert
    test_generator._render_part_mock_method.assert_called_once()
    # Check that section_data passed to _render_part contains shared_tracks
    call_args_render = test_generator._render_part_mock_method.call_args[0]
    assert "shared_tracks" in call_args_render[0]

    mock_get_part_override.assert_called_once_with(None, "TestSection", test_generator.part_name)

    # process_one checks (should not be called as no groove_profile_path or part_specific_humanize_params)
    mock_load_groove_profile.assert_not_called()
    mock_apply_groove_pretty.assert_not_called()
    mock_apply_humanization.assert_not_called()

    # final_process checks
    mock_humanize_apply_base.assert_called_once_with(rendered_part, None)
    mock_apply_swing.assert_called_once_with(rendered_part, 0.0, 8) # Default swing
    mock_apply_envelope.assert_called_once_with(rendered_part, 0, 4, 1.0) # Default intensity, q_length

    assert result_part is rendered_part # Part is modified in-place
    mock_logger.info.assert_any_call(f"Rendering part for section: 'TestSection' with overrides: None")


@patch(f"{BGP_PATH}.apply_envelope")
@patch(f"{BGP_PATH}.apply_swing")
@patch(f"{BGP_PATH}.humanize_apply")
@patch(f"{BGP_PATH}.get_part_override")
def test_compose_dict_parts_with_hand_specific_swing_overrides(
    mock_get_part_override: Mock,
    mock_humanize_apply_base: Mock,
    mock_apply_swing: Mock,
    mock_apply_envelope: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    part_rh = stream.Part(id="rh_part"); part_rh.append(note.Note("C5"))
    part_lh = stream.Part(id="lh_part"); part_lh.append(note.Note("C3"))
    test_generator._render_part_mock_method.return_value = {"rh": part_rh, "lh": part_lh, "other": stream.Part()}

    # Mock overrides with hand-specific swing
    mock_override_obj = MockPydanticModel(swing_ratio=0.5, swing_ratio_rh=0.7, swing_ratio_lh=0.6)
    mock_get_part_override.return_value = mock_override_obj
    
    # Mock OverrideModelType instance to pass to compose
    mock_overrides_root = Mock()

    # Act
    result_parts = test_generator.compose(section_data=default_section_data.copy(), overrides_root=mock_overrides_root)

    # Assert
    mock_get_part_override.assert_called_once_with(mock_overrides_root, "TestSection", test_generator.part_name)
    
    # Check calls for each part
    assert mock_humanize_apply_base.call_count == 3
    mock_humanize_apply_base.assert_any_call(part_rh, None)
    mock_humanize_apply_base.assert_any_call(part_lh, None)

    assert mock_apply_swing.call_count == 3
    mock_apply_swing.assert_any_call(part_rh, 0.7, 8) # swing_ratio_rh
    mock_apply_swing.assert_any_call(part_lh, 0.6, 8) # swing_ratio_lh
    mock_apply_swing.assert_any_call(result_parts["other"], 0.5, 8) # fallback to main swing_ratio

    assert mock_apply_envelope.call_count == 3
    mock_apply_envelope.assert_any_call(part_rh, 0, 4, 1.0)
    mock_apply_envelope.assert_any_call(part_lh, 0, 4, 1.0)

    assert isinstance(result_parts, dict)
    assert result_parts["rh"] is part_rh
    assert result_parts["lh"] is part_lh
    mock_logger.info.assert_any_call(f"Rendering part for section: 'TestSection' with overrides: {mock_override_obj.model_dump()}")


@patch(f"{BGP_PATH}.apply_swing")
@patch(f"{BGP_PATH}.get_part_override") # Other mocks not needed for this specific test
def test_compose_swing_ratio_from_section_params(
    mock_get_part_override: Mock,
    mock_apply_swing: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock # To satisfy fixture dependencies
):
    # Arrange
    rendered_part = stream.Part(id="s_part"); rendered_part.append(note.Note("D4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    mock_get_part_override.return_value = None # No overrides

    section_data = default_section_data.copy()
    section_data["part_params"] = {"swing_ratio": 0.65}

    # Act
    test_generator.compose(section_data=section_data)

    # Assert
    mock_apply_swing.assert_called_once_with(rendered_part, 0.65, 8)


@patch(f"{BGP_PATH}.apply_swing")
@patch(f"{BGP_PATH}.get_part_override")
def test_compose_variable_swing_list(
    mock_get_part_override: Mock,
    mock_apply_swing: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock,
):
    rendered_part = stream.Part(id="vs_part"); rendered_part.append(note.Note("E4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    mock_get_part_override.return_value = None

    section_data = default_section_data.copy()
    section_data["part_params"] = {"swing_ratio": [0.55, 0.45]}

    test_generator.compose(section_data=section_data)

    mock_apply_swing.assert_called_once_with(rendered_part, [0.55, 0.45], subdiv=8)


@patch(f"{BGP_PATH}.apply_groove_pretty")
@patch(f"{BGP_PATH}.load_groove_profile")
@patch(f"{BGP_PATH}.get_part_override") # Other mocks not needed
def test_compose_with_groove_profile(
    mock_get_part_override: Mock,
    mock_load_groove_profile: Mock,
    mock_apply_groove_pretty: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    rendered_part = stream.Part(id="g_part"); rendered_part.append(note.Note("E4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    mock_get_part_override.return_value = None
    
    mock_groove_profile = Mock()
    mock_load_groove_profile.return_value = mock_groove_profile
    groove_path = "path/to/groove.json"

    # Act
    test_generator.compose(section_data=default_section_data.copy(), groove_profile_path=groove_path)

    # Assert
    mock_load_groove_profile.assert_called_once_with(groove_path)
    mock_apply_groove_pretty.assert_called_once_with(rendered_part, mock_groove_profile)
    mock_logger.info.assert_any_call(f"Applied groove from '{groove_path}' to {test_generator.part_name}.")


@patch(f"{BGP_PATH}.apply_humanization_to_part")
@patch(f"{BGP_PATH}.get_part_override") # Other mocks not needed
def test_compose_with_part_specific_humanization(
    mock_get_part_override: Mock,
    mock_apply_humanization: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    rendered_part = stream.Part(id="h_part"); rendered_part.append(note.Note("F4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    mock_get_part_override.return_value = None

    humanize_params = {"enable": True, "template_name": "custom_template", "custom_params": {"timing": 0.1}}
    
    # Act
    test_generator.compose(section_data=default_section_data.copy(), part_specific_humanize_params=humanize_params)

    # Assert
    mock_apply_humanization.assert_called_once_with(
        rendered_part, template_name="custom_template", custom_params={"timing": 0.1}
    )
    mock_logger.info.assert_any_call(f"Applied final touch humanization (template: custom_template) to {test_generator.part_name}.")


@patch(f"{BGP_PATH}.apply_envelope")
@patch(f"{BGP_PATH}.get_part_override") # Other mocks not needed
def test_compose_intensity_scaling(
    mock_get_part_override: Mock,
    mock_apply_envelope: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    rendered_part = stream.Part(id="i_part"); rendered_part.append(note.Note("A4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    mock_get_part_override.return_value = None

    section_data = default_section_data.copy()
    section_data["musical_intent"]["intensity"] = "high" # Scale should be 1.1

    # Act
    test_generator.compose(section_data=section_data)

    # Assert
    mock_apply_envelope.assert_called_once_with(rendered_part, 0, 4, 1.1)


def test_compose_shared_tracks_merging(
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock # To satisfy fixture dependencies
):
    # Arrange
    initial_shared = {"key1": "val1"}
    section_data = default_section_data.copy()
    section_data["shared_tracks"] = initial_shared.copy()
    
    compose_shared = {"key2": "val2"}

    # Act
    with patch.object(test_generator, '_render_part_mock_method') as mock_render:
        mock_render.return_value = stream.Part() # Needs to return a part
        # Patch other dependencies of final_process to avoid errors if not relevant to this test focus
        with patch(f"{BGP_PATH}.humanize_apply"), \
             patch(f"{BGP_PATH}.apply_swing"), \
             patch(f"{BGP_PATH}.apply_envelope"), \
             patch(f"{BGP_PATH}.get_part_override", return_value=None):
            test_generator.compose(section_data=section_data, shared_tracks=compose_shared)

    # Assert
    mock_render.assert_called_once()
    called_section_data = mock_render.call_args[0][0]
    assert "shared_tracks" in called_section_data
    assert called_section_data["shared_tracks"]["key1"] == "val1"
    assert called_section_data["shared_tracks"]["key2"] == "val2"


@patch(f"{BGP_PATH}.get_part_override", return_value=None) # Mocks to simplify
def test_compose_render_part_invalid_return(
    mock_get_override: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    test_generator._render_part_mock_method.return_value = None # Invalid return

    # Act
    result = test_generator.compose(section_data=default_section_data.copy())

    # Assert
    assert isinstance(result, stream.Part)
    assert not result.flatten().notes # Should be empty
    assert result.id == test_generator.part_name
    mock_logger.error.assert_called_once_with(
        f"_render_part for {test_generator.part_name} did not return a valid stream.Part or dict. Returning empty part."
    )


@patch(f"{BGP_PATH}.apply_groove_pretty", side_effect=Exception("Groove error"))
@patch(f"{BGP_PATH}.load_groove_profile", return_value=Mock())
@patch(f"{BGP_PATH}.get_part_override", return_value=None)
def test_compose_error_in_apply_groove(
    mock_get_override: Mock,
    mock_load_profile: Mock,
    mock_apply_groove: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    rendered_part = stream.Part(id="eg_part"); rendered_part.append(note.Note("C4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    
    # Act
    # Patch other dependencies of final_process
    with patch(f"{BGP_PATH}.humanize_apply"), \
         patch(f"{BGP_PATH}.apply_swing"), \
         patch(f"{BGP_PATH}.apply_envelope"):
        test_generator.compose(section_data=default_section_data.copy(), groove_profile_path="dummy_path")

    # Assert
    mock_logger.error.assert_any_call(
        f"Error applying groove to {test_generator.part_name}: Groove error", exc_info=True
    )


@patch(f"{BGP_PATH}.apply_humanization_to_part", side_effect=Exception("Humanize error"))
@patch(f"{BGP_PATH}.get_part_override", return_value=None)
def test_compose_error_in_apply_humanization(
    mock_get_override: Mock,
    mock_apply_humanize: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock
):
    # Arrange
    rendered_part = stream.Part(id="eh_part"); rendered_part.append(note.Note("D4"))
    test_generator._render_part_mock_method.return_value = rendered_part
    humanize_params = {"enable": True}

    # Act
    # Patch other dependencies of final_process
    with patch(f"{BGP_PATH}.humanize_apply"), \
         patch(f"{BGP_PATH}.apply_swing"), \
         patch(f"{BGP_PATH}.apply_envelope"):
        test_generator.compose(section_data=default_section_data.copy(), part_specific_humanize_params=humanize_params)

    # Assert
    mock_logger.error.assert_any_call(
        f"Error during final touch humanization for {test_generator.part_name}: Humanize error", exc_info=True
    )


@patch(f"{BGP_PATH}.apply_humanization_to_part")
@patch(f"{BGP_PATH}.apply_groove_pretty")
@patch(f"{BGP_PATH}.load_groove_profile")
@patch(f"{BGP_PATH}.get_part_override", return_value=None)
def test_compose_no_notes_skips_processing(
    mock_get_override: Mock,
    mock_load_profile: Mock,
    mock_apply_groove: Mock,
    mock_apply_humanize: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock # To satisfy fixture dependencies
):
    # Arrange
    rendered_part = stream.Part(id="no_notes_part") # No notes
    test_generator._render_part_mock_method.return_value = rendered_part
    
    humanize_params = {"enable": True}
    groove_path = "dummy_path"

    # Act
    # Patch other dependencies of final_process
    with patch(f"{BGP_PATH}.humanize_apply"), \
         patch(f"{BGP_PATH}.apply_swing"), \
         patch(f"{BGP_PATH}.apply_envelope"):
        test_generator.compose(
            section_data=default_section_data.copy(),
            groove_profile_path=groove_path,
            part_specific_humanize_params=humanize_params
        )

    # Assert
    mock_load_profile.assert_not_called() # Skipped because no notes
    mock_apply_groove.assert_not_called() # Skipped because no notes
    mock_apply_humanize.assert_not_called() # Skipped because no notes


@patch(f"{BGP_PATH}.apply_offset_profile")
@patch(f"{BGP_PATH}.apply_envelope")
@patch(f"{BGP_PATH}.apply_swing")
@patch(f"{BGP_PATH}.humanize_apply")
@patch(f"{BGP_PATH}.get_part_override")
def test_compose_offset_profile_hand_specific(
    mock_get_override: Mock,
    mock_humanize_apply: Mock,
    mock_apply_swing: Mock,
    mock_apply_envelope: Mock,
    mock_apply_offset: Mock,
    test_generator: ConcreteTestGenerator,
    default_section_data: dict,
    mock_logger: Mock,
):
    part_rh = stream.Part(id="rh_p"); part_rh.append(note.Note("C5"))
    part_lh = stream.Part(id="lh_p"); part_lh.append(note.Note("C3"))
    test_generator._render_part_mock_method.return_value = {"rh": part_rh, "lh": part_lh, "other": stream.Part()}

    ov_obj = MockPydanticModel(offset_profile="main", offset_profile_rh="rh_prof", offset_profile_lh="lh_prof")
    mock_get_override.return_value = ov_obj

    result = test_generator.compose(section_data=default_section_data.copy(), overrides_root=Mock())

    mock_apply_offset.assert_any_call(part_rh, "rh_prof")
    mock_apply_offset.assert_any_call(part_lh, "lh_prof")
    mock_apply_offset.assert_any_call(result["other"], "main")
    assert mock_apply_offset.call_count == 3


def test_extra_cc_not_duplicated(test_generator: ConcreteTestGenerator, default_section_data: dict, mock_logger: Mock) -> None:
    """Ensure _auto_tone_shape does not duplicate CC events across calls."""
    part1 = stream.Part(id="cc1"); part1.append(note.Note("C4"))
    test_generator._render_part_mock_method.return_value = part1
    out1 = test_generator.compose(section_data=default_section_data.copy())
    len1 = len(getattr(out1, "extra_cc", []))

    part2 = stream.Part(id="cc2"); part2.append(note.Note("C4"))
    test_generator._render_part_mock_method.return_value = part2
    out2 = test_generator.compose(section_data=default_section_data.copy())
    len2 = len(getattr(out2, "extra_cc", []))

    assert len1 == len2
