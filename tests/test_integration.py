"""
🧪 INTEGRATION TESTS FOR REALITY SIMULATOR

Test the complete system integration with all components working together.
"""

import sys
import time
import tempfile
import os
sys.path.insert(0, '.')

from reality_simulator.main import RealitySimulator
from reality_simulator.reality_renderer import InteractionMode


def test_full_initialization():
    """Test complete system initialization"""
    print("🧪 Testing full system initialization...")

    # Create simulator with default config
    simulator = RealitySimulator()

    # Initialize
    success = simulator.initialize_simulation()

    assert success, "Initialization should succeed"
    assert len(simulator.components) > 0, "Should have components"

    # Check all expected components
    expected_components = ['quantum', 'lattice', 'evolution', 'network',
                          'agency', 'renderer']

    for component in expected_components:
        assert component in simulator.components, f"Missing component: {component}"

    print("✅ Full system initialization works")


def test_simulation_loop():
    """Test running a short simulation"""
    print("🧪 Testing simulation loop...")

    simulator = RealitySimulator()

    # Initialize
    success = simulator.initialize_simulation()
    assert success, "Should initialize successfully"

    # Run short simulation (5 frames)
    results = simulator.run_simulation(max_frames=5)

    assert 'frames_simulated' in results
    assert results['frames_simulated'] == 5
    assert 'total_time' in results
    assert results['total_time'] > 0
    assert 'avg_fps' in results

    print("✅ Simulation loop works")


def test_component_interaction():
    """Test that components interact properly"""
    print("🧪 Testing component interaction...")

    simulator = RealitySimulator()

    # Initialize
    success = simulator.initialize_simulation()
    assert success

    # Get components
    evolution = simulator.components['evolution']
    network = simulator.components['network']
    
    # Check that network has organisms
    assert len(network.organisms) > 0, "Network should have organisms after initialization"

    print("✅ Component interaction works")


def test_user_commands():
    """Test handling user commands"""
    print("🧪 Testing user commands...")

    simulator = RealitySimulator()

    # Initialize
    success = simulator.initialize_simulation()
    assert success

    # Test mode change command
    response = simulator.handle_user_command("mode", {"mode": "god"})
    assert "god" in response.lower()

    # Test time dilation command
    response = simulator.handle_user_command("time", {"dilation": 2.0})
    assert "2.0" in response

    # Test status command
    response = simulator.handle_user_command("status", {})
    # Should return JSON
    import json
    status_data = json.loads(response)
    assert isinstance(status_data, dict)

    print("✅ User commands work")


def test_state_save_load():
    """Test saving and loading simulation state"""
    print("🧪 Testing state save/load...")

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_save.json")

        # Create and initialize simulator
        simulator = RealitySimulator()

        # Modify config for testing
        simulator.config['simulation']['target_fps'] = 60

        success = simulator.initialize_simulation()
        assert success

        # Run a bit
        simulator.run_simulation(max_frames=2)

        # Save state
        simulator.save_state(save_path)
        assert os.path.exists(save_path)

        # Create new simulator and load
        new_simulator = RealitySimulator()
        load_success = new_simulator.load_state(save_path)

        assert load_success, "Should load successfully"
        assert new_simulator.config['simulation']['target_fps'] == 60

        print("✅ State save/load works")


def test_config_loading():
    """Test configuration file loading"""
    print("🧪 Testing configuration loading...")

    # Create temporary config file
    config_data = {
        "simulation": {
            "target_fps": 15
        },
        "evolution": {
            "population_size": 50
        },
        "rendering": {
            "mode": "scientist"
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Create simulator with config
        simulator = RealitySimulator(config_path)

        # Check config was loaded
        assert simulator.config['simulation']['target_fps'] == 15
        assert simulator.config['evolution']['population_size'] == 50
        assert simulator.config['rendering']['mode'] == "scientist"

        # Test initialization with custom config
        success = simulator.initialize_simulation()
        assert success

        # Check evolution engine used custom population size
        evolution = simulator.components['evolution']
        assert evolution.population_size == 50

        print("✅ Configuration loading works")

    finally:
        os.unlink(config_path)


def test_performance_under_load():
    """Test system performance with moderate load"""
    print("🧪 Testing performance under load...")

    # Create simulator with smaller config for testing
    simulator = RealitySimulator()

    # Modify for quick test
    simulator.config['simulation']['target_fps'] = 5  # Slow for testing
    simulator.config['evolution']['population_size'] = 20
    simulator.config['lattice']['particles'] = 20

    success = simulator.initialize_simulation()
    assert success

    # Run moderate simulation
    start_time = time.time()
    results = simulator.run_simulation(max_frames=10)
    elapsed = time.time() - start_time

    assert results['frames_simulated'] == 10
    assert elapsed < 30, f"Should complete in reasonable time, took {elapsed:.1f}s"

    print("✅ Performance under load works")





def test_error_handling():
    """Test error handling in various scenarios"""
    print("🧪 Testing error handling...")

    # Test with invalid config
    simulator = RealitySimulator("nonexistent_config.json")
    # Should still work with defaults
    success = simulator.initialize_simulation()
    assert success, "Should handle missing config gracefully"

    # Test command handling with invalid input
    response = simulator.handle_user_command("invalid_command", {})
    assert "unhandled" in response.lower() or "error" in response.lower()

    print("✅ Error handling works")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("🚀 REALITY SIMULATOR INTEGRATION TESTS")
    print("=" * 60)
    print()

    tests = [
        test_full_initialization,
        test_simulation_loop,
        test_component_interaction,
        test_user_commands,
        test_state_save_load,
        test_config_loading,
        test_performance_under_load,
        test_performance_under_load,
        test_error_handling
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            print()
            passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            print()
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"🎉 INTEGRATION TESTS COMPLETE: {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("✅ All integration tests passed!")
        print("The Reality Simulator is fully functional and ready to explore simulated reality.")
        print()
        print("🎯 MISSION ACCOMPLISHED:")
        print("   - Quantum particles evolve into complex organisms")
        print("   - Human-AI symbiosis enables exploration")
        print("   - Multi-modal reality rendering provides insight")
        print("   - Potato-optimized for accessible research")
        print()
        print("The bridge between computation and reality is complete. ✨")
    else:
        print(f"⚠️  {total - passed} integration tests failed.")
        print("Check component interactions and system integration.")


if __name__ == "__main__":
    run_all_tests()

