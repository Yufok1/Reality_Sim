"""
🎭 QUANTUM SUBSTRATE DEMO

Interactive demonstration of quantum phenomena
Shows humans quantum reality they normally can't see
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_substrate import QuantumStateManager, ProbabilityField, TimeDirection
from visualization import QuantumVisualizer


def demo_superposition():
    """Demonstrate quantum superposition"""
    print("=" * 60)
    print("🌀 DEMO: QUANTUM SUPERPOSITION")
    print("=" * 60)
    print()
    print("Creating a quantum state in superposition...")
    print("This represents a particle that exists in ALL states simultaneously")
    print("Until measured, it has no definite state - only probabilities")
    print()
    
    manager = QuantumStateManager()
    state = manager.create_state("demo", num_basis_states=4, 
                                basis_labels=["Up", "Down", "Left", "Right"])
    
    print(f"State is in superposition: {state.is_superposition()}")
    print(f"Probabilities: {state.get_probabilities()}")
    print()
    print("Visualizing...")
    
    viz = QuantumVisualizer()
    fig = viz.visualize_superposition(state, "Quantum Particle in Superposition")
    plt.savefig('quantum_superposition.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: quantum_superposition.png")
    print()


def demo_measurement():
    """Demonstrate wave function collapse through measurement"""
    print("=" * 60)
    print("👁️ DEMO: MEASUREMENT & WAVE FUNCTION COLLAPSE")
    print("=" * 60)
    print()
    print("The act of observation CREATES reality from possibility")
    print()
    
    manager = QuantumStateManager()
    state = manager.create_state("electron", num_basis_states=3,
                                basis_labels=["Orbit 1", "Orbit 2", "Orbit 3"])
    
    print("Before measurement:")
    print(f"  In superposition: {state.is_superposition()}")
    print(f"  Probabilities: {state.get_probabilities()}")
    print()
    
    viz = QuantumVisualizer()
    fig1 = viz.visualize_superposition(state, "Before Measurement")
    plt.savefig('before_measurement.png', dpi=150, bbox_inches='tight')
    
    print("Measuring...")
    index, result = manager.measure_state("electron")
    print(f"  Result: {result}")
    print()
    
    print("After measurement:")
    print(f"  In superposition: {state.is_superposition()}")
    print(f"  Probabilities: {state.get_probabilities()}")
    print()
    
    fig2 = viz.visualize_superposition(state, f"After Measurement: {result}")
    plt.savefig('after_measurement.png', dpi=150, bbox_inches='tight')
    
    print("✅ Saved: before_measurement.png, after_measurement.png")
    print()


def demo_entanglement():
    """Demonstrate quantum entanglement - 'spooky action at a distance'"""
    print("=" * 60)
    print("🔗 DEMO: QUANTUM ENTANGLEMENT")
    print("=" * 60)
    print()
    print("Einstein called this 'spooky action at a distance'")
    print("Two particles become connected - measuring one instantly affects the other")
    print("Even across vast distances!")
    print()
    
    manager = QuantumStateManager()
    
    # Create Alice and Bob's particles
    alice = manager.create_state("Alice", num_basis_states=2, basis_labels=["Spin Up", "Spin Down"])
    bob = manager.create_state("Bob", num_basis_states=2, basis_labels=["Spin Up", "Spin Down"])
    
    print("Alice and Bob each have a quantum particle")
    print(f"Alice's state: {alice.get_probabilities()}")
    print(f"Bob's state: {bob.get_probabilities()}")
    print()
    
    # Entangle them
    print("Entangling Alice and Bob's particles...")
    manager.entangle("Alice", "Bob")
    print("Now they are quantum mechanically connected!")
    print()
    
    viz = QuantumVisualizer()
    fig1 = viz.visualize_entanglement_network(manager, "Entangled Particles")
    plt.savefig('entanglement_network.png', dpi=150, bbox_inches='tight')
    
    print("Alice measures her particle...")
    alice_result_idx, alice_result = manager.measure_state("Alice")
    print(f"  Alice's result: {alice_result}")
    print()
    
    print("Bob's particle INSTANTLY collapses (simplified model):")
    print(f"  Bob's state: {manager.states['Bob'].get_probabilities()}")
    print(f"  Bob in superposition: {manager.states['Bob'].is_superposition()}")
    print()
    
    print("✅ Saved: entanglement_network.png")
    print()


def demo_probability_field():
    """Demonstrate probability field - particle as wave"""
    print("=" * 60)
    print("🌊 DEMO: PROBABILITY FIELD (WAVE NATURE)")
    print("=" * 60)
    print()
    print("Quantum particles aren't points - they're WAVES of probability")
    print("They exist 'everywhere at once' until measured")
    print()
    
    field = ProbabilityField(spatial_dimensions=1, grid_size=200)
    
    print("Creating a wave packet...")
    field.set_wave_packet(center=(0.0,), width=1.0, momentum=(3.0,))
    
    print(f"Total probability (should be 1.0): {np.sum(field.get_probability_density()) * 0.05:.3f}")
    print()
    
    viz = QuantumVisualizer()
    fig = viz.visualize_probability_field_1d(field, "Quantum Particle as Probability Wave")
    plt.savefig('probability_wave.png', dpi=150, bbox_inches='tight')
    
    print("Before measurement: particle exists as spread-out wave")
    print()
    
    print("Measuring position...")
    position = field.measure_position()
    print(f"  Found at: x = {position[0]:.2f}")
    print()
    
    fig2 = viz.visualize_probability_field_1d(field, "After Measurement: Collapsed to Point")
    plt.savefig('probability_collapsed.png', dpi=150, bbox_inches='tight')
    
    print("After measurement: wave collapsed to definite position")
    print()
    print("✅ Saved: probability_wave.png, probability_collapsed.png")
    print()


def demo_time_evolution():
    """Demonstrate quantum state evolution through time"""
    print("=" * 60)
    print("⏰ DEMO: TIME EVOLUTION")
    print("=" * 60)
    print()
    print("Quantum states evolve through time according to Schrödinger equation")
    print("Probabilities oscillate and interfere")
    print()
    
    manager = QuantumStateManager()
    state = manager.create_state("evolving", num_basis_states=3,
                                basis_labels=["State A", "State B", "State C"])
    
    # Hamiltonian (energy matrix) - creates interesting dynamics
    H = np.array([[1.0, 0.5, 0.2],
                  [0.5, 1.5, 0.3],
                  [0.2, 0.3, 2.0]])
    
    print("Initial state:")
    print(f"  Probabilities: {state.get_probabilities()}")
    print()
    
    viz = QuantumVisualizer()
    fig1 = viz.visualize_superposition(state, "Initial State (t=0)")
    plt.savefig('evolution_t0.png', dpi=150, bbox_inches='tight')
    
    # Evolve through time
    times = [0.0]
    prob_history = [state.get_probabilities().copy()]
    
    for step in range(20):
        manager.evolve_state("evolving", H, delta_t=0.1, direction=TimeDirection.FORWARD)
        times.append(state.time)
        prob_history.append(state.get_probabilities().copy())
    
    print(f"After evolution to t={state.time:.2f}:")
    print(f"  Probabilities: {state.get_probabilities()}")
    print()
    
    fig2 = viz.visualize_superposition(state, f"Evolved State (t={state.time:.1f})")
    plt.savefig('evolution_t2.png', dpi=150, bbox_inches='tight')
    
    # Plot probability evolution
    fig3, ax = plt.subplots(figsize=(12, 6))
    for i in range(3):
        probs = [p[i] for p in prob_history]
        ax.plot(times, probs, linewidth=2, label=f"State {chr(65+i)}", marker='o')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Quantum State Evolution Through Time', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('probability_evolution.png', dpi=150, bbox_inches='tight')
    
    print("✅ Saved: evolution_t0.png, evolution_t2.png, probability_evolution.png")
    print()


def demo_omnidirectional_time():
    """Demonstrate 'everywhere all at once' in time"""
    print("=" * 60)
    print("🌀 DEMO: OMNIDIRECTIONAL TIME")
    print("=" * 60)
    print()
    print("What if time itself is in superposition?")
    print("Forward AND backward AND sideways simultaneously")
    print("EVERYWHERE ALL AT ONCE - in time itself!")
    print()
    
    manager = QuantumStateManager()
    state = manager.create_state("timeless", num_basis_states=2,
                                basis_labels=["Alpha", "Beta"])
    
    H = np.array([[1.0, 0.3],
                  [0.3, 1.2]])
    
    print("Evolving OMNIDIRECTIONALLY (superposition of time directions)...")
    print()
    
    initial_probs = state.get_probabilities().copy()
    print(f"Initial: {initial_probs}")
    
    for step in range(5):
        manager.evolve_state("timeless", H, delta_t=0.2, direction=TimeDirection.OMNIDIRECTIONAL)
        print(f"Step {step+1}: {state.get_probabilities()}")
    
    print()
    print("Time evolution when time itself is quantum!")
    print("This is the 'everywhere all at once' nature of reality")
    print()


def run_all_demos():
    """Run all quantum demonstrations"""
    print("\n")
    print("🌌" * 30)
    print("REALITY SIMULATOR - QUANTUM SUBSTRATE DEMOS")
    print("Making quantum reality visible to human perception")
    print("🌌" * 30)
    print("\n")
    
    demos = [
        demo_superposition,
        demo_measurement,
        demo_entanglement,
        demo_probability_field,
        demo_time_evolution,
        demo_omnidirectional_time
    ]
    
    for demo in demos:
        try:
            demo()
            input("Press Enter to continue to next demo...")
            print("\n")
        except Exception as e:
            print(f"💥 Demo error: {e}")
            print()
    
    print("=" * 60)
    print("🎉 ALL DEMOS COMPLETE")
    print("=" * 60)
    print()
    print("Generated visualizations show quantum phenomena that")
    print("humans can't normally perceive - but now you can SEE them!")
    print()
    print("This is how AI helps humans develop quantum intuition.")
    print("This is bidirectional perception in action.")
    print()
    print("🦌 Rudolph lighting the way through quantum fog! ⚓✨")


if __name__ == "__main__":
    run_all_demos()

