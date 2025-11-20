"""
🧪 TESTS FOR REALITY RENDERER

Test visualization modes, interaction paradigms, and rendering pipeline.
"""

import numpy as np
import sys
import time
sys.path.insert(0, '.')

from reality_simulator.reality_renderer import (
    RealityRenderer, InteractionMode, RenderState, VisualizationConfig,
    create_reality_renderer, render_text_interface
)


def test_renderer_initialization():
    """Test renderer initialization and configuration"""
    print("[TEST] Testing renderer initialization...")

    config = VisualizationConfig(resolution=(1280, 720), frame_rate=60.0)
    renderer = RealityRenderer(config)

    assert renderer.config.resolution == (1280, 720)
    assert renderer.config.frame_rate == 60.0
    assert renderer.state.mode == InteractionMode.OBSERVER  # Default
    assert len(renderer.visualization_modules) > 0

    print("[TEST] Renderer initialization works")


def test_interaction_modes():
    """Test switching between interaction modes"""
    print("[TEST] Testing interaction modes...")

    renderer = RealityRenderer()

    # Test mode switching
    renderer.set_interaction_mode(InteractionMode.GOD, "Testing god mode")
    assert renderer.state.mode == InteractionMode.GOD
    assert "god_overview" in renderer.state.active_visualizations

    renderer.set_interaction_mode(InteractionMode.PARTICIPANT, "Testing participant mode")
    assert renderer.state.mode == InteractionMode.PARTICIPANT
    assert renderer.state.user_position is not None

    renderer.set_interaction_mode(InteractionMode.SCIENTIST, "Testing scientist mode")
    assert renderer.state.mode == InteractionMode.SCIENTIST

    print("[TEST] Interaction modes work")


def test_rendering_pipeline():
    """Test the rendering pipeline"""
    print("[TEST] Testing rendering pipeline...")

    renderer = RealityRenderer()

    # Test frame rendering
    frame_data = renderer.render_frame()

    assert "frame_number" in frame_data
    assert "timestamp" in frame_data
    assert "mode" in frame_data
    assert "visualizations" in frame_data
    assert frame_data["frame_number"] == 0

    # Test second frame
    frame_data2 = renderer.render_frame()
    assert frame_data2["frame_number"] == 1

    print("[TEST] Rendering pipeline works")


def test_mode_specific_rendering():
    """Test mode-specific rendering features"""
    print("[TEST] Testing mode-specific rendering...")

    renderer = RealityRenderer()

    # Mock simulation data
    mock_data = {
        "quantum": {"states": 100},
        "lattice": {"particles": 50, "cpu_usage": 45.0, "ram_usage": 2.1},
        "evolution": {"generation": 10, "population_size": 20, "best_fitness": 0.8, "avg_fitness": 0.6},
        "network": {"organisms": 15, "connections": 25, "stability": 0.7, "connectivity": 0.8},
        "agency": {"mode": "manual_only", "performance": {"total_decisions": 5}}
    }

    # Test god mode rendering
    renderer.set_interaction_mode(InteractionMode.GOD)
    god_data = renderer._render_god_mode(mock_data)
    assert "universe_overview" in god_data
    assert "control_panels" in god_data

    # Test observer mode rendering
    renderer.set_interaction_mode(InteractionMode.OBSERVER)
    observer_data = renderer._render_observer_mode(mock_data)
    assert "scientific_metrics" in observer_data
    assert "data_visualizations" in observer_data

    # Test participant mode rendering
    renderer.set_interaction_mode(InteractionMode.PARTICIPANT)
    participant_data = renderer._render_participant_mode(mock_data)
    assert "user_position" in participant_data
    assert "sensory_input" in participant_data

    # Test scientist mode rendering
    renderer.set_interaction_mode(InteractionMode.SCIENTIST)
    scientist_data = renderer._render_scientist_mode(mock_data)
    assert "experimental_tools" in scientist_data

    print("[TEST] Mode-specific rendering works")


def test_user_input_handling():
    """Test user input handling in different modes"""
    print("[TEST] Testing user input handling...")

    renderer = RealityRenderer()

    # Test god mode input
    renderer.set_interaction_mode(InteractionMode.GOD)
    response = renderer.handle_user_input("time_control", {"dilation": 2.0})
    assert response["mode"] == "god"
    assert renderer.state.time_dilation == 2.0

    # Test participant mode input
    renderer.set_interaction_mode(InteractionMode.PARTICIPANT)
    response = renderer.handle_user_input("movement", {"direction": [1, 0, 0], "speed": 0.5})
    assert response["mode"] == "participant"
    assert response["position_updated"] is not None

    print("[TEST] User input handling works")


def test_performance_monitoring():
    """Test performance monitoring"""
    print("[TEST] Testing performance monitoring...")

    renderer = RealityRenderer()

    # Render a few frames
    for _ in range(3):
        renderer.render_frame()
        time.sleep(0.01)  # Small delay

    stats = renderer.get_performance_stats()

    assert "frames_rendered" in stats
    assert "elapsed_time" in stats
    assert "average_fps" in stats
    assert stats["frames_rendered"] == 3
    assert stats["elapsed_time"] > 0

    print("[TEST] Performance monitoring works")


def test_visualization_modules():
    """Test individual visualization modules"""
    print("[TEST] Testing visualization modules...")

    renderer = RealityRenderer()

    # Test that all expected modules are present
    expected_modules = [
        "quantum_field", "particle_cloud", "evolution_tree",
        "network_graph", "agency_flow",
        "performance_monitor", "god_overview"
    ]

    for module_name in expected_modules:
        assert module_name in renderer.visualization_modules

    # Test rendering a specific module
    mock_data = {"network": {"organisms": 10, "connections": 15}}
    mock_state = RenderState()
    mock_config = VisualizationConfig()

    network_viz = renderer.visualization_modules["network_graph"]
    result = network_viz.render(mock_data, mock_state, mock_config)

    assert "nodes" in result
    assert "edges" in result
    assert result["nodes"] == 10

    print("[TEST] Visualization modules work")


def test_utility_functions():
    """Test utility functions"""
    print("[TEST] Testing utility functions...")

    # Test renderer creation
    renderer = create_reality_renderer(InteractionMode.SCIENTIST)
    assert renderer.state.mode == InteractionMode.SCIENTIST

    # Test text interface rendering
    mock_data = {
        "lattice": {"particles": 42, "cpu_usage": 25.0, "ram_usage": 1.5},
        "evolution": {"generation": 5, "population_size": 30, "best_fitness": 0.75},
        "network": {"organisms": 20, "connections": 35, "stability": 0.65}
    }

    text_output = render_text_interface(renderer, mock_data)

    assert isinstance(text_output, str)
    assert len(text_output) > 100
    assert "REALITY SIMULATOR" in text_output
    assert "SCIENTIST MODE" in text_output

    print("[TEST] Utility functions work")


def test_component_injection():
    """Test injecting simulation components"""
    print("[TEST] Testing component injection...")

    renderer = RealityRenderer()

    # Mock components
    class MockComponent:
        pass

    mock_quantum = MockComponent()
    mock_lattice = MockComponent()
    mock_evolution = MockComponent()

    # Inject components
    renderer.inject_simulation_components(
        quantum_manager=mock_quantum,
        lattice=mock_lattice,
        evolution_engine=mock_evolution
    )

    assert renderer.quantum_manager is mock_quantum
    assert renderer.lattice is mock_lattice
    assert renderer.evolution_engine is mock_evolution

    print("[TEST] Component injection works")


def test_nearby_entity_detection():
    """Test nearby entity detection for participant mode"""
    print("[TEST] Testing nearby entity detection...")

    renderer = RealityRenderer()

    # Create mock lattice with particles
    class MockParticle:
        def __init__(self, pos):
            self.position = np.array(pos)
            self.charge = 1.0
            self.fitness = 0.5

    mock_particles = [
        MockParticle([0, 0, 0]),
        MockParticle([2, 0, 0]),
        MockParticle([10, 0, 0])  # Too far
    ]

    renderer.lattice = (mock_particles, None, None)  # Mock lattice tuple

    # Test nearby detection
    position = np.array([0, 0, 0])
    nearby = renderer._find_nearby_entities(position, radius=5.0)

    assert len(nearby) == 2  # Two particles within radius 5
    assert nearby[0]["distance"] == 0.0  # First particle at origin
    assert nearby[1]["distance"] == 2.0  # Second particle 2 units away

    print("[TEST] Nearby entity detection works")


def run_all_tests():
    """Run all reality renderer tests"""
    print("=" * 60)
    print("=" * 60)
    print("[TEST] REALITY RENDERER TESTS")
    print("=" * 60)
    print()

    tests = [
        test_renderer_initialization,
        test_interaction_modes,
        test_rendering_pipeline,
        test_mode_specific_rendering,
        test_user_input_handling,
        test_performance_monitoring,
        test_visualization_modules,
        test_utility_functions,
        test_component_injection,
        test_nearby_entity_detection
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            print()
            passed += 1
        except Exception as e:
            print(f"[TEST] TEST FAILED: {e}")
            print()
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("=" * 60)
    print(f"[TEST] TESTS COMPLETE: {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("[TEST] All reality renderer tests passed!")
        print("The visualization system is ready to render simulated reality.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check implementation.")


if __name__ == "__main__":
    run_all_tests()

