"""
🌀 QUANTUM SUBSTRATE (Layer 0)

The fundamental reality layer - quantum mechanics simulation

Features:
- Superposition: All states exist simultaneously
- Entanglement: Non-local correlations
- Measurement: Wave function collapse
- Time Evolution: Forward, backward, and omnidirectional
"""

import numpy as np
import time
import sys
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TimeDirection(Enum):
    """Direction of time evolution"""
    FORWARD = "forward"
    BACKWARD = "backward"
    OMNIDIRECTIONAL = "omnidirectional"


@dataclass
class QuantumState:
    """
    Represents a quantum state in superposition
    
    Attributes:
        amplitudes: Complex probability amplitudes for each basis state
        basis_labels: Labels for each basis state
        time: Current time coordinate
        entangled_with: List of entangled state IDs
    """
    amplitudes: np.ndarray  # Complex probability amplitudes
    basis_labels: List[str]
    time: float = 0.0
    entangled_with: List[str] = None
    
    def __post_init__(self):
        if self.entangled_with is None:
            self.entangled_with = []
        # Normalize amplitudes
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
    
    def get_probabilities(self) -> np.ndarray:
        """Calculate measurement probabilities from amplitudes"""
        return np.abs(self.amplitudes)**2
    
    def measure(self) -> Tuple[int, str]:
        """
        Collapse wave function through measurement
        Returns: (index, basis_label) of measured state
        """
        probabilities = self.get_probabilities()
        measured_index = np.random.choice(len(probabilities), p=probabilities)
        
        # Collapse to measured state
        self.amplitudes = np.zeros_like(self.amplitudes)
        self.amplitudes[measured_index] = 1.0
        
        return measured_index, self.basis_labels[measured_index]
    
    def is_superposition(self) -> bool:
        """Check if state is in superposition (vs collapsed)"""
        probabilities = self.get_probabilities()
        return np.sum(probabilities > 0.01) > 1  # More than one significant state


class QuantumStateManager:
    """
    Manages quantum states and their evolution
    
    The heart of the quantum substrate - handles superposition,
    entanglement, and measurement.
    """
    
    def __init__(self, fitness_weights: Optional[Dict[str, float]] = None,
                 performance_thresholds: Optional[Dict[str, float]] = None):
        self.states: Dict[str, QuantumState] = {}
        self.entanglement_network: Dict[str, List[str]] = {}
        self.measurement_history: List[Dict] = []
        
        # Fitness tracking for adaptive pruning
        self.state_measurement_count: Dict[str, int] = {}
        self.state_creation_time: Dict[str, float] = {}
        self.estimated_memory_per_state = 400  # bytes (rough estimate: amplitudes, labels, metadata)
        self.total_states_memory: float = 0.0
        
        # Fitness weights (defaults if not provided)
        self.fitness_weights = fitness_weights or {
            'entanglement': 0.3,
            'superposition': 0.25,
            'measurements': 0.25,
            'entropy': 0.2
        }
        
        # Performance thresholds (defaults if not provided)
        self.performance_thresholds = performance_thresholds or {
            'memory_percentage': 5.0,  # % of RAM
            'iteration_time_ms': 10.0,  # milliseconds
            'fitness_std_threshold': 0.3,  # std dev of fitness scores
            'min_fitness_to_keep': 0.1  # minimum fitness to keep
        }
    
    def create_state(self, state_id: str, num_basis_states: int, 
                     basis_labels: Optional[List[str]] = None) -> QuantumState:
        """
        Create a new quantum state in equal superposition
        
        Args:
            state_id: Unique identifier for this state
            num_basis_states: Number of basis states
            basis_labels: Optional labels for basis states
        
        Returns:
            The created QuantumState
        """
        if basis_labels is None:
            basis_labels = [f"state_{i}" for i in range(num_basis_states)]
        
        # Equal superposition: all amplitudes equal
        amplitudes = np.ones(num_basis_states, dtype=complex) / np.sqrt(num_basis_states)
        
        state = QuantumState(amplitudes=amplitudes, basis_labels=basis_labels)
        self.states[state_id] = state
        self.entanglement_network[state_id] = []
        
        # Track creation time and initialize measurement count for fitness calculation
        self.state_creation_time[state_id] = time.time()
        self.state_measurement_count[state_id] = 0
        self.total_states_memory += self.estimated_memory_per_state
        
        return state
    
    def entangle(self, state_id_1: str, state_id_2: str):
        """
        Create quantum entanglement between two states
        
        Entangled states are correlated - measuring one affects the other
        instantaneously, regardless of distance (non-locality)
        """
        if state_id_1 not in self.states or state_id_2 not in self.states:
            raise ValueError("Both states must exist to entangle")
        
        # Bidirectional entanglement
        self.states[state_id_1].entangled_with.append(state_id_2)
        self.states[state_id_2].entangled_with.append(state_id_1)
        
        self.entanglement_network[state_id_1].append(state_id_2)
        self.entanglement_network[state_id_2].append(state_id_1)
    
    def measure_state(self, state_id: str) -> Tuple[int, str]:
        """
        Measure a quantum state, collapsing its wave function
        
        If the state is entangled, this affects entangled partners
        """
        if state_id not in self.states:
            raise ValueError(f"State {state_id} does not exist")
        
        state = self.states[state_id]
        measured_index, measured_label = state.measure()
        
        # Increment measurement count for fitness calculation
        if state_id in self.state_measurement_count:
            self.state_measurement_count[state_id] += 1
        else:
            self.state_measurement_count[state_id] = 1
        
        # Record measurement
        self.measurement_history.append({
            'state_id': state_id,
            'time': state.time,
            'result': measured_label,
            'index': measured_index
        })
        
        # Collapse entangled states (simplified model)
        for entangled_id in state.entangled_with:
            if entangled_id in self.states:
                entangled_state = self.states[entangled_id]
                if entangled_state.is_superposition():
                    # Correlated collapse (simplified - real entanglement is more complex)
                    entangled_state.measure()
        
        return measured_index, measured_label
    
    def evolve_state(self, state_id: str, hamiltonian: np.ndarray, 
                     delta_t: float, direction: TimeDirection = TimeDirection.FORWARD):
        """
        Evolve quantum state through time using Schrödinger equation
        
        Args:
            state_id: State to evolve
            hamiltonian: Hamiltonian operator (energy matrix)
            delta_t: Time step
            direction: Direction of time evolution
        """
        if state_id not in self.states:
            raise ValueError(f"State {state_id} does not exist")
        
        state = self.states[state_id]
        
        if direction == TimeDirection.OMNIDIRECTIONAL:
            # Superposition of different time directions
            # (Conceptual - demonstrates "everywhere all at once" in time)
            forward = self._apply_time_evolution(state.amplitudes, hamiltonian, delta_t)
            backward = self._apply_time_evolution(state.amplitudes, hamiltonian, -delta_t)
            # Mix forward and backward evolution
            state.amplitudes = (forward + backward) / np.sqrt(2)
            state.time += delta_t * 0.5  # Average time progression
        else:
            # Standard unidirectional evolution
            dt = delta_t if direction == TimeDirection.FORWARD else -delta_t
            state.amplitudes = self._apply_time_evolution(state.amplitudes, hamiltonian, dt)
            state.time += dt
    
    def _apply_time_evolution(self, amplitudes: np.ndarray, hamiltonian: np.ndarray, 
                              delta_t: float) -> np.ndarray:
        """
        Apply time evolution operator: ψ(t+dt) = exp(-iHdt/ℏ) ψ(t)
        
        Uses matrix exponentiation for exact evolution
        """
        hbar = 1.0  # Natural units
        evolution_operator = self._matrix_exp(-1j * hamiltonian * delta_t / hbar)
        return evolution_operator @ amplitudes
    
    def _matrix_exp(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix exponential using eigendecomposition"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        exp_eigenvalues = np.exp(eigenvalues)
        return eigenvectors @ np.diag(exp_eigenvalues) @ np.linalg.inv(eigenvectors)
    
    def get_entanglement_density(self) -> float:
        """
        Calculate overall entanglement density of the system
        Returns: Average number of entangled connections per state
        """
        if not self.states:
            return 0.0
        total_connections = sum(len(connections) for connections in self.entanglement_network.values())
        return total_connections / len(self.states)
    
    def get_superposition_count(self) -> int:
        """Count how many states are currently in superposition"""
        return sum(1 for state in self.states.values() if state.is_superposition())
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        probabilities_safe = probabilities + epsilon
        entropy = -np.sum(probabilities * np.log2(probabilities_safe))
        return entropy
    
    def calculate_state_fitness(self, state_id: str) -> float:
        """
        Calculate fitness score for a quantum state
        
        Fitness is based on multiple factors:
        - Entanglement count (more connections = more valuable)
        - Superposition status (active states preferred)
        - Measurement frequency (frequently used states are important)
        - Entropy (higher entropy = more "quantum interesting")
        
        Returns: Fitness score [0.0, 1.0]
        """
        if state_id not in self.states:
            return 0.0
        
        state = self.states[state_id]
        weights = self.fitness_weights
        
        # Factor 1: Entanglement count (normalized to max 10)
        entanglement_count = len(state.entangled_with)
        entanglement_score = min(1.0, entanglement_count / 10.0)
        
        # Factor 2: Superposition status
        superposition_score = 1.0 if state.is_superposition() else 0.0
        
        # Factor 3: Measurement frequency (normalized to max 20)
        measurement_count = self.state_measurement_count.get(state_id, 0)
        measurement_score = min(1.0, measurement_count / 20.0)
        
        # Factor 4: Entropy of probability distribution (normalized to max ~2.0 for 4 states)
        probabilities = state.get_probabilities()
        entropy = self._calculate_entropy(probabilities)
        entropy_score = min(1.0, entropy / 2.0)  # Max entropy for 4 equal states ~2.0
        
        # Weighted combination
        fitness = (
            weights['entanglement'] * entanglement_score +
            weights['superposition'] * superposition_score +
            weights['measurements'] * measurement_score +
            weights['entropy'] * entropy_score
        )
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, fitness))
    
    def should_prune(self) -> Tuple[bool, Optional[int], str]:
        """
        Determine if pruning is needed based on performance metrics
        
        Checks multiple performance indicators:
        1. Memory pressure (total memory vs available RAM)
        2. Iteration overhead (time to iterate over all states)
        3. Fitness distribution (std dev of fitness scores)
        
        Returns: (should_prune: bool, optimal_count: Optional[int], reason: str)
        """
        if not self.states:
            return (False, None, "No states to prune")
        
        current_count = len(self.states)
        thresholds = self.performance_thresholds
        
        # Check 1: Memory pressure
        try:
            available_ram_gb = psutil.virtual_memory().available / (1024**3)  # GB
            total_memory_mb = (self.total_states_memory / (1024**2))  # MB
            memory_percentage = (total_memory_mb / (available_ram_gb * 1024)) * 100
            
            if memory_percentage > thresholds['memory_percentage']:
                # Calculate optimal count to bring memory usage to threshold
                target_memory_mb = (available_ram_gb * 1024) * (thresholds['memory_percentage'] / 100)
                optimal_count = int(current_count * (target_memory_mb / total_memory_mb))
                optimal_count = max(1, min(optimal_count, current_count - 1))  # Keep at least 1, don't exceed current
                return (True, optimal_count, 
                       f"Memory usage {memory_percentage:.2f}% exceeds threshold {thresholds['memory_percentage']}%")
        except Exception:
            pass  # If psutil fails, skip memory check
        
        # Check 2: Iteration overhead (time to calculate fitness for all states)
        try:
            start_time = time.time()
            fitness_scores = [self.calculate_state_fitness(state_id) for state_id in self.states.keys()]
            iteration_time_ms = (time.time() - start_time) * 1000
            
            if iteration_time_ms > thresholds['iteration_time_ms']:
                # Calculate optimal count to bring iteration time to threshold
                optimal_count = int(current_count * (thresholds['iteration_time_ms'] / iteration_time_ms))
                optimal_count = max(1, min(optimal_count, current_count - 1))
                return (True, optimal_count,
                       f"Iteration time {iteration_time_ms:.2f}ms exceeds threshold {thresholds['iteration_time_ms']}ms")
        except Exception:
            pass  # If timing fails, skip time check
        
        # Check 3: Fitness distribution (too many low-fitness states)
        try:
            fitness_scores = [self.calculate_state_fitness(state_id) for state_id in self.states.keys()]
            if len(fitness_scores) > 1:
                fitness_std = float(np.std(fitness_scores))
                
                if fitness_std < thresholds['fitness_std_threshold']:
                    # Too many low-fitness states (flat distribution)
                    # Remove 20% of lowest-fitness states
                    optimal_count = max(1, int(current_count * 0.8))
                    return (True, optimal_count,
                           f"Fitness distribution too flat (std={fitness_std:.3f} < {thresholds['fitness_std_threshold']})")
        except Exception:
            pass  # If calculation fails, skip fitness check
        
        # System is performing optimally - no pruning needed
        return (False, None, "System performing optimally - no pruning needed")

    def set_pruning_aggressiveness(self, aggressiveness: float):
        """
        Set pruning aggressiveness (0.0 = conservative, 1.0 = aggressive)

        Adjusts performance thresholds to be more or less strict.
        Higher aggressiveness = lower thresholds = more frequent pruning.
        """
        # Clamp to valid range
        aggressiveness = max(0.0, min(1.0, aggressiveness))

        # Base thresholds (from config)
        base_thresholds = {
            'memory_percentage': 5.0,
            'iteration_time_ms': 10.0,
            'fitness_std_threshold': 0.3,
            'min_fitness_to_keep': 0.1
        }

        # Scale thresholds based on aggressiveness
        # Higher aggressiveness = lower thresholds (more pruning)
        scaling_factor = 1.0 - (aggressiveness * 0.8)  # 0.2 to 1.0 range

        self.performance_thresholds = {
            'memory_percentage': base_thresholds['memory_percentage'] * scaling_factor,
            'iteration_time_ms': base_thresholds['iteration_time_ms'] * scaling_factor,
            'fitness_std_threshold': base_thresholds['fitness_std_threshold'] * scaling_factor,
            'min_fitness_to_keep': base_thresholds['min_fitness_to_keep']  # Don't scale this one
        }

    def get_pruning_aggressiveness(self) -> float:
        """Get current pruning aggressiveness level"""
        # This is a rough approximation - in practice we'd need to store the current value
        # For now, return 0.5 as default
        return 0.5
    
    def prune_low_fitness_states(self, target_count: Optional[int] = None) -> int:
        """
        Prune low-fitness quantum states to maintain optimal performance
        
        Args:
            target_count: Target number of states to keep. If None, calculates from performance metrics.
            
        Returns:
            Number of states removed
        """
        if not self.states:
            return 0
        
        current_count = len(self.states)
        
        # Determine target count
        if target_count is None:
            should_prune, optimal_count, reason = self.should_prune()
            if not should_prune or optimal_count is None:
                return 0  # No pruning needed
            target_count = optimal_count
        else:
            target_count = max(1, min(target_count, current_count - 1))  # Keep at least 1, don't exceed current
        
        if target_count >= current_count:
            return 0  # No pruning needed
        
        # Calculate fitness for all states
        state_fitness = {}
        for state_id in self.states.keys():
            state_fitness[state_id] = self.calculate_state_fitness(state_id)
        
        # Sort states by fitness (descending)
        sorted_states = sorted(state_fitness.items(), key=lambda x: x[1], reverse=True)
        
        # Determine how many to keep
        # Conservative: Always keep top 20% by fitness (safety buffer)
        min_fitness_to_keep = self.performance_thresholds.get('min_fitness_to_keep', 0.1)
        keep_count = max(
            target_count,  # Target from performance metrics
            int(current_count * 0.2),  # Always keep top 20%
            len([f for _, f in sorted_states if f > min_fitness_to_keep])  # Keep all above threshold
        )
        keep_count = min(keep_count, current_count)  # Don't exceed current count
        
        # Determine states to keep
        states_to_keep = {state_id for state_id, fitness in sorted_states[:keep_count]}
        states_to_remove = set(self.states.keys()) - states_to_keep
        
        # Prune states (remove in reverse order to avoid index issues)
        removed_count = 0
        for state_id in states_to_remove:
            state = self.states[state_id]
            
            # Clean up entangled state references
            for entangled_id in state.entangled_with:
                if entangled_id in self.states:
                    partner_state = self.states[entangled_id]
                    if state_id in partner_state.entangled_with:
                        partner_state.entangled_with.remove(state_id)
                if entangled_id in self.entanglement_network:
                    if state_id in self.entanglement_network[entangled_id]:
                        self.entanglement_network[entangled_id].remove(state_id)
            
            # Remove from dictionaries
            del self.states[state_id]
            if state_id in self.entanglement_network:
                del self.entanglement_network[state_id]
            if state_id in self.state_measurement_count:
                del self.state_measurement_count[state_id]
            if state_id in self.state_creation_time:
                del self.state_creation_time[state_id]
            
            # Update memory tracking
            self.total_states_memory -= self.estimated_memory_per_state
            removed_count += 1
        
        return removed_count


class ProbabilityField:
    """
    Represents the probability field across space
    
    In quantum mechanics, particles don't have definite positions -
    they exist as probability waves across all of space.
    """
    
    def __init__(self, spatial_dimensions: int = 3, grid_size: int = 50):
        self.dimensions = spatial_dimensions
        self.grid_size = grid_size
        self.field = self._initialize_field()
    
    def _initialize_field(self) -> np.ndarray:
        """Create spatial grid for probability field"""
        shape = tuple([self.grid_size] * self.dimensions)
        return np.zeros(shape, dtype=complex)
    
    def set_wave_packet(self, center: Tuple[float, ...], width: float, momentum: Tuple[float, ...]):
        """
        Create a Gaussian wave packet
        
        Args:
            center: Center position of wave packet
            width: Spatial width (uncertainty in position)
            momentum: Average momentum (affects wavelength)
        """
        # Create coordinate grids
        coords = [np.linspace(-5, 5, self.grid_size) for _ in range(self.dimensions)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        # Gaussian envelope
        gaussian = np.ones_like(grids[0], dtype=complex)
        for i, (grid, c, p) in enumerate(zip(grids, center, momentum)):
            gaussian *= np.exp(-((grid - c)**2) / (2 * width**2))
            gaussian *= np.exp(1j * p * grid)  # Plane wave component
        
        # Normalize wave function for proper probability density
        # For integral to be 1.0: sum(|ψ|²) * dx = 1, so sum(|ψ|²) = 1/dx
        # Grid spans [-5, 5] in each dimension, so dx = 10.0 / grid_size
        grid_spacing = 10.0 / self.grid_size
        norm = np.sqrt(np.sum(np.abs(gaussian)**2) * (grid_spacing ** len(center)))
        if norm > 0:
            self.field = gaussian / norm
        else:
            self.field = gaussian
    
    def get_probability_density(self) -> np.ndarray:
        """Calculate |ψ|² probability density"""
        return np.abs(self.field)**2
    
    def measure_position(self) -> Tuple[float, ...]:
        """
        Measure position, collapsing wave function
        Returns: Position coordinates
        """
        prob_density = self.get_probability_density()
        prob_density_flat = prob_density.flatten()
        prob_density_flat /= np.sum(prob_density_flat)
        
        # Sample from probability distribution
        index = np.random.choice(len(prob_density_flat), p=prob_density_flat)
        
        # Convert flat index to multi-dimensional coordinates
        coords = np.unravel_index(index, prob_density.shape)
        
        # Map to actual spatial coordinates
        spatial_coords = tuple(
            -5 + (c / self.grid_size) * 10 for c in coords
        )
        
        # Collapse wave function to measured position
        self.field = np.zeros_like(self.field)
        self.field[coords] = 1.0
        
        return spatial_coords


# Module-level docstring for humans reading the code
"""
🎨 HUMAN IMAGINATION → 🔗 AI CONNECTION

This module demonstrates quantum superposition, entanglement, and measurement.

Key insights:
- Reality exists in superposition until observed
- Observation (measurement) creates definite reality from possibility
- Entanglement means particles remain connected regardless of distance
- Time can flow in multiple directions (omnidirectional evolution)

This is the foundation layer - everything else builds on quantum substrate.
"""

