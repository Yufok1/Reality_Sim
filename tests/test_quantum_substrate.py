"""
🧪 TESTS FOR QUANTUM SUBSTRATE

Verify that quantum mechanics behaves correctly
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from reality_simulator.quantum_substrate import (
    QuantumState, QuantumStateManager, ProbabilityField, 
    TimeDirection
)


def test_superposition_creation():
    """Test that states start in equal superposition"""
    print("🧪 Testing superposition creation...")
    
    manager = QuantumStateManager()
    state = manager.create_state("test", num_basis_states=4)
    
    # Check equal superposition
    probabilities = state.get_probabilities()
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    
    assert np.allclose(probabilities, expected), "Should start in equal superposition"
    assert state.is_superposition(), "Should be in superposition"
    
    print("✅ Superposition created correctly")


def test_measurement_collapse():
    """Test that measurement collapses superposition"""
    print("🧪 Testing measurement collapse...")
    
    manager = QuantumStateManager()
    state = manager.create_state("test", num_basis_states=3)
    
    assert state.is_superposition(), "Should start in superposition"
    
    # Measure
    index, label = manager.measure_state("test")
    
    assert not state.is_superposition(), "Should collapse after measurement"
    assert state.get_probabilities()[index] > 0.99, "Measured state should have probability ~1"
    
    print(f"✅ Measurement collapsed to: {label}")


def test_entanglement():
    """Test quantum entanglement between states"""
    print("🧪 Testing entanglement...")
    
    manager = QuantumStateManager()
    state1 = manager.create_state("alice", num_basis_states=2, basis_labels=["up", "down"])
    state2 = manager.create_state("bob", num_basis_states=2, basis_labels=["up", "down"])
    
    # Entangle Alice and Bob
    manager.entangle("alice", "bob")
    
    assert "bob" in state1.entangled_with, "Alice should be entangled with Bob"
    assert "alice" in state2.entangled_with, "Bob should be entangled with Alice"
    
    # Measure Alice
    alice_result_idx, alice_result = manager.measure_state("alice")
    
    # Bob should also collapse (simplified model)
    assert not manager.states["bob"].is_superposition(), "Bob should collapse when Alice measured"
    
    print(f"✅ Entanglement working - Alice: {alice_result}")


def test_time_evolution():
    """Test quantum state evolution through time"""
    print("🧪 Testing time evolution...")
    
    manager = QuantumStateManager()
    state = manager.create_state("evolving", num_basis_states=2)
    
    # Simple Hamiltonian (energy matrix)
    H = np.array([[1.0, 0.5], 
                  [0.5, 1.0]])
    
    initial_amplitudes = state.amplitudes.copy()
    
    # Evolve forward in time
    manager.evolve_state("evolving", H, delta_t=0.1, direction=TimeDirection.FORWARD)
    
    # State should change
    assert not np.allclose(state.amplitudes, initial_amplitudes), "State should evolve"
    
    # Probability should still be normalized
    assert np.isclose(np.sum(state.get_probabilities()), 1.0), "Probabilities should sum to 1"
    
    print("✅ Time evolution working")


def test_omnidirectional_time():
    """Test omnidirectional time evolution (everywhere all at once in time)"""
    print("🧪 Testing omnidirectional time...")
    
    manager = QuantumStateManager()
    state = manager.create_state("timeless", num_basis_states=2)
    
    H = np.array([[1.0, 0.2], 
                  [0.2, 1.0]])
    
    # Evolve omnidirectionally (superposition of time directions)
    manager.evolve_state("timeless", H, delta_t=0.1, direction=TimeDirection.OMNIDIRECTIONAL)
    
    # State should evolve but differently than pure forward/backward
    assert state.is_superposition(), "Should remain in superposition"
    
    print("✅ Omnidirectional time evolution working")


def test_probability_field():
    """Test probability field and wave packets"""
    print("🧪 Testing probability field...")
    
    field = ProbabilityField(spatial_dimensions=1, grid_size=100)
    
    # Create wave packet at origin
    field.set_wave_packet(center=(0.0,), width=1.0, momentum=(2.0,))
    
    # Check probability density integrates to 1
    prob_density = field.get_probability_density()
    total_prob = np.sum(prob_density) * (10.0 / 100)  # Grid spacing
    
    assert np.isclose(total_prob, 1.0, atol=0.1), "Probability should integrate to 1"
    
    # Measure position
    position = field.measure_position()
    
    print(f"✅ Probability field working - measured position: {position}")


def test_entanglement_density():
    """Test network entanglement metrics"""
    print("🧪 Testing entanglement density...")
    
    manager = QuantumStateManager()
    
    # Create network of states
    for i in range(5):
        manager.create_state(f"particle_{i}", num_basis_states=2)
    
    # Entangle in a chain
    for i in range(4):
        manager.entangle(f"particle_{i}", f"particle_{i+1}")
    
    density = manager.get_entanglement_density()
    
    # Each particle has ~2 connections (except endpoints with 1)
    # Average should be 1.6
    expected_density = 8 / 5  # Total connections / number of particles
    
    assert np.isclose(density, expected_density), f"Entanglement density should be {expected_density}"
    
    print(f"✅ Entanglement density: {density}")


def run_all_tests():
    """Run all quantum substrate tests"""
    print("=" * 60)
    print("🌀 QUANTUM SUBSTRATE TESTS")
    print("=" * 60)
    print()
    
    tests = [
        test_superposition_creation,
        test_measurement_collapse,
        test_entanglement,
        test_time_evolution,
        test_omnidirectional_time,
        test_probability_field,
        test_entanglement_density
    ]
    
    for test in tests:
        try:
            test()
            print()
        except AssertionError as e:
            print(f"❌ TEST FAILED: {e}")
            print()
        except Exception as e:
            print(f"💥 ERROR: {e}")
            print()
    
    print("=" * 60)
    print("🎉 QUANTUM SUBSTRATE TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

