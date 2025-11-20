"""
🌀 SUBATOMIC LATTICE (Layer 1)

The building blocks of reality - particles that carry genetic information
through quantum approximation and entropy-based state management.

Features:
- Particle class with allelic properties (quantum numbers → genetic traits)
- AllелicMapper for trait conversion
- FixedPointAttractor for stable configurations
- FieldInteraction for particle dynamics
- QuantumStateApproximator with entropy pruning (99.9% state reduction)
- ResourceMonitor for CPU/RAM tracking and auto-scaling
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import os


class ParticleType(Enum):
    """Types of subatomic particles with their quantum properties"""
    QUARK = {"charge": 1/3, "spin": 1/2, "baryon_number": 1/3, "color_charge": True}
    LEPTON = {"charge": -1, "spin": 1/2, "baryon_number": 0, "color_charge": False}
    BOSON = {"charge": 0, "spin": 0, "baryon_number": 0, "color_charge": False}


@dataclass
class AllelicProperties:
    """Genetic traits derived from particle quantum numbers"""
    dominance: float  # From particle charge (0.0 = recessive, 1.0 = dominant)
    strength: float   # From particle spin (0.0 = weak, 1.0 = strong)
    stability: float  # From particle mass/energy (0.0 = unstable, 1.0 = stable)
    interaction: float # From particle color/weak charge (0.0 = isolated, 1.0 = social)

    def to_genetic_code(self) -> str:
        """Convert to 4-base genetic code (A/T/G/C)"""
        code = ""
        code += "A" if self.dominance > 0.5 else "T"
        code += "G" if self.strength > 0.5 else "C"
        code += "A" if self.stability > 0.5 else "T"
        code += "G" if self.interaction > 0.5 else "T"  # T for low interaction, not C
        return code

    @classmethod
    def from_genetic_code(cls, code: str) -> 'AllelicProperties':
        """Convert from genetic code back to properties"""
        if len(code) != 4:
            raise ValueError("Genetic code must be 4 bases")

        # Decode base pairs
        dominance = 1.0 if code[0] == "A" else 0.0
        strength = 1.0 if code[1] == "G" else 0.0
        stability = 1.0 if code[2] == "A" else 0.0
        interaction = 1.0 if code[3] == "G" else 0.0  # T or C both map to 0.0

        return cls(dominance=dominance, strength=strength,
                  stability=stability, interaction=interaction)


@dataclass
class Particle:
    """
    Subatomic particle with quantum properties that map to genetic traits

    Each particle carries "genetic information" through its quantum numbers,
    which get mapped to allelic properties for evolution.
    """
    particle_type: ParticleType
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    charge: float = 0.0
    spin: float = 0.0
    mass: float = 1.0
    energy: float = 0.0

    # Derived genetic properties
    alleles: AllelicProperties = field(init=False)

    def __post_init__(self):
        # Derive genetic traits from quantum properties
        self._calculate_allelic_properties()

    def _calculate_allelic_properties(self):
        """Map quantum properties to genetic traits"""
        # Dominance from charge (absolute value, normalized)
        dominance = min(1.0, abs(self.charge))

        # Strength from spin magnitude
        strength = min(1.0, abs(self.spin) / 2.0)  # Max spin 2 for bosons

        # Stability from mass/energy ratio (higher = more stable)
        stability = min(1.0, self.mass / (self.energy + 1.0))

        # Interaction from particle type properties
        interaction = 0.0
        if self.particle_type == ParticleType.QUARK:
            interaction = 1.0  # Strong nuclear force
        elif self.particle_type == ParticleType.LEPTON:
            interaction = 0.5  # Weak nuclear + electromagnetic
        else:  # BOSON
            interaction = 0.8  # Force carrier

        self.alleles = AllelicProperties(
            dominance=dominance,
            strength=strength,
            stability=stability,
            interaction=interaction
        )

    def update_energy(self):
        """Update energy based on relativistic mass-energy equivalence"""
        # E = sqrt((pc)^2 + (mc^2)^2) approximation
        momentum_magnitude = np.linalg.norm(self.momentum)
        rest_energy = self.mass * (3e8)**2  # c^2 approximation
        kinetic_energy = momentum_magnitude * 3e8  # pc approximation
        self.energy = np.sqrt(rest_energy**2 + kinetic_energy**2)

        # Recalculate alleles since energy changed
        self._calculate_allelic_properties()

    def get_genetic_code(self) -> str:
        """Get genetic code for this particle"""
        return self.alleles.to_genetic_code()


class AllelicMapper:
    """
    Maps between particle quantum properties and genetic traits

    Handles the conversion between subatomic physics and evolutionary biology
    """

    @staticmethod
    def particle_to_allele(particle: Particle) -> AllelicProperties:
        """Extract genetic traits from a particle"""
        return particle.alleles

    @staticmethod
    def allele_to_particle(alleles: AllelicProperties,
                          particle_type: ParticleType = ParticleType.QUARK) -> Particle:
        """Create a particle from genetic traits (reverse mapping)"""
        # Convert back to quantum properties
        charge = alleles.dominance * (1/3 if particle_type == ParticleType.QUARK else -1)
        spin = alleles.strength * 2.0
        mass = alleles.stability * 10.0  # Arbitrary scaling
        energy = mass * (3e8)**2

        return Particle(
            particle_type=particle_type,
            charge=charge,
            spin=spin,
            mass=mass,
            energy=energy
        )

    @staticmethod
    def crossover(parent1: AllelicProperties, parent2: AllelicProperties) -> AllelicProperties:
        """Genetic crossover between two allele sets"""
        # Simple single-point crossover
        dominance = parent1.dominance if np.random.random() > 0.5 else parent2.dominance
        strength = parent1.strength if np.random.random() > 0.5 else parent2.strength
        stability = parent1.stability if np.random.random() > 0.5 else parent2.stability
        interaction = parent1.interaction if np.random.random() > 0.5 else parent2.interaction

        return AllelicProperties(dominance=dominance, strength=strength,
                               stability=stability, interaction=interaction)

    @staticmethod
    def mutate(alleles: AllelicProperties, rate: float = 0.01) -> AllelicProperties:
        """Apply random mutations to alleles"""
        if np.random.random() < rate:
            alleles.dominance = np.clip(alleles.dominance + np.random.normal(0, 0.1), 0, 1)
        if np.random.random() < rate:
            alleles.strength = np.clip(alleles.strength + np.random.normal(0, 0.1), 0, 1)
        if np.random.random() < rate:
            alleles.stability = np.clip(alleles.stability + np.random.normal(0, 0.1), 0, 1)
        if np.random.random() < rate:
            alleles.interaction = np.clip(alleles.interaction + np.random.normal(0, 0.1), 0, 1)

        return alleles


class FixedPointAttractor:
    """
    Finds stable configurations in particle space using gradient descent

    Attractors represent "stable genetic configurations" that evolution
    tends toward - the equivalent of evolutionary stable strategies.
    """

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def find_attractor(self, particles: List[Particle],
                      energy_function: callable = None) -> List[Particle]:
        """
        Find stable configuration using gradient descent

        Args:
            particles: Current particle configuration
            energy_function: Function that calculates total system energy

        Returns:
            Stable particle configuration
        """
        if energy_function is None:
            energy_function = self._default_energy_function

        # Extract positions for optimization
        positions = np.array([p.position for p in particles])
        positions_flat = positions.flatten()

        # Gradient descent to minimize energy
        for iteration in range(self.max_iterations):
            energy = energy_function(particles)

            # Calculate gradients (simplified finite differences)
            gradients = np.zeros_like(positions_flat)

            for i in range(len(positions_flat)):
                # Perturb position
                positions_test = positions_flat.copy()
                positions_test[i] += self.tolerance

                # Calculate energy difference
                particles_test = self._update_positions(particles, positions_test)
                energy_test = energy_function(particles_test)
                gradients[i] = (energy_test - energy) / self.tolerance

            # Update positions
            learning_rate = 0.01
            positions_flat -= learning_rate * gradients

            # Check convergence
            if np.max(np.abs(gradients)) < self.tolerance:
                break

        # Update particles with final positions
        return self._update_positions(particles, positions_flat)

    def _default_energy_function(self, particles: List[Particle]) -> float:
        """Default energy function based on particle interactions"""
        total_energy = 0.0

        for i, p1 in enumerate(particles):
            # Self-energy
            total_energy += p1.energy

            # Interaction energy with other particles
            for j, p2 in enumerate(particles):
                if i != j:
                    distance = np.linalg.norm(p1.position - p2.position)
                    if distance > 0:
                        # Coulomb-like interaction
                        interaction = (p1.charge * p2.charge) / (distance + 1.0)
                        total_energy += interaction

        return total_energy

    def _update_positions(self, particles: List[Particle],
                         positions_flat: np.ndarray) -> List[Particle]:
        """Update particle positions from flattened array"""
        positions = positions_flat.reshape(len(particles), -1)

        for i, particle in enumerate(particles):
            particle.position = positions[i]
            particle.update_energy()  # Energy depends on position/momentum

        return particles


class FieldInteraction:
    """
    Handles forces and interactions between particles

    Models electromagnetic, strong, weak, and gravitational forces
    using simplified approximations for computational efficiency.
    """

    def __init__(self, gravitational_constant: float = 1e-10,
                 electromagnetic_strength: float = 1.0,
                 strong_force_range: float = 1.0):
        self.G = gravitational_constant
        self.em_strength = electromagnetic_strength
        self.strong_range = strong_force_range

    def calculate_forces(self, particles: List[Particle]) -> List[np.ndarray]:
        """
        Calculate forces on each particle from all others

        Returns list of force vectors (one per particle)
        """
        forces = [np.zeros(3) for _ in particles]

        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles):
                if i != j:
                    force = self._calculate_pair_force(p1, p2)
                    forces[i] += force

        return forces

    def _calculate_pair_force(self, p1: Particle, p2: Particle) -> np.ndarray:
        """Calculate force between two particles"""
        displacement = p2.position - p1.position
        distance = np.linalg.norm(displacement)

        if distance == 0:
            return np.zeros(3)

        direction = displacement / distance

        force = np.zeros(3)

        # Gravitational force
        if self.G > 0:
            gravitational = self.G * p1.mass * p2.mass / (distance**2 + 1e-10)
            force += gravitational * direction

        # Electromagnetic force (Coulomb)
        if self.em_strength > 0:
            electromagnetic = self.em_strength * p1.charge * p2.charge / (distance**2 + 1e-10)
            force += electromagnetic * direction

        # Strong nuclear force (simplified Yukawa potential)
        if p1.particle_type == ParticleType.QUARK and p2.particle_type == ParticleType.QUARK:
            strong_force = -np.exp(-distance / self.strong_range) / distance**2
            force += strong_force * direction

        return force

    def update_particles(self, particles: List[Particle], dt: float = 0.01):
        """
        Update particle positions and momenta using forces

        Uses simple symplectic Euler integration
        """
        forces = self.calculate_forces(particles)

        for particle, force in zip(particles, forces):
            # F = ma, a = F/m
            acceleration = force / (particle.mass + 1e-10)  # Avoid division by zero

            # Update momentum and position
            particle.momentum += acceleration * dt
            particle.position += (particle.momentum / particle.mass) * dt

            # Update energy based on new state
            particle.update_energy()


class QuantumStateApproximator:
    """
    Approximates quantum behavior with entropy-based state pruning

    Instead of simulating full quantum superposition, we use probabilistic
    models with massive state reduction (99.9% target).

    NOTE: max_states limits active particles for performance. GUI shows configured count,
    but only max_states particles remain active after entropy pruning.
    """

    def __init__(self, entropy_threshold: float = 0.8,
                 max_states: int = 1000):
        self.entropy_threshold = entropy_threshold
        self.max_states = max_states
        self.pruned_states = {}  # Cache for pruned state calculations

    def approximate_superposition(self, particles: List[Particle]) -> List[Particle]:
        """
        Approximate quantum superposition by selecting high-entropy states

        Uses entropy as a proxy for "quantum interestingness"
        """
        # Calculate entropy for each particle
        entropies = []
        for particle in particles:
            entropy = self._calculate_particle_entropy(particle)
            entropies.append(entropy)

        # Select particles above entropy threshold
        selected_indices = [i for i, e in enumerate(entropies)
                          if e >= self.entropy_threshold]

        # If too many, take the highest entropy ones
        if len(selected_indices) > self.max_states:
            entropy_pairs = sorted(zip(entropies, selected_indices),
                                 key=lambda x: x[0], reverse=True)
            selected_indices = [idx for _, idx in entropy_pairs[:self.max_states]]

        # Return selected particles (pruned state space)
        selected_particles = [particles[i] for i in selected_indices]

        # Update pruning statistics (stored for monitoring, not printed)
        original_count = len(particles)
        pruned_count = len(selected_particles)
        reduction_ratio = 1.0 - (pruned_count / original_count) if original_count > 0 else 0
        
        # Store stats for monitoring (no debug spam)
        self.last_pruning_stats = {
            'original': original_count,
            'pruned': pruned_count,
            'reduction': reduction_ratio,
            'entropy_range': (min(entropies), max(entropies)) if entropies else (0, 0)
        }

        return selected_particles

    def _calculate_particle_entropy(self, particle: Particle) -> float:
        """
        Calculate entropy for a particle based on its quantum properties

        Higher entropy = more "quantum interesting" = less likely to be pruned
        """
        # Entropy from position uncertainty (spread)
        position_entropy = -np.log(np.linalg.norm(particle.position) + 1e-10)

        # Entropy from momentum uncertainty
        momentum_entropy = -np.log(np.linalg.norm(particle.momentum) + 1e-10)

        # Entropy from quantum number diversity
        quantum_entropy = 0.0
        if particle.charge != 0:
            quantum_entropy += 1.0
        if particle.spin != 0:
            quantum_entropy += 1.0
        if particle.mass > 1.0:
            quantum_entropy += 1.0

        # Combine entropies
        total_entropy = (position_entropy + momentum_entropy + quantum_entropy) / 3.0

        # Normalize to [0, 1]
        return min(1.0, max(0.0, (total_entropy + 5.0) / 10.0))  # Shift and scale


class ResourceMonitor:
    """
    Monitors CPU, RAM, and other resources during simulation

    Provides real-time feedback and auto-scaling suggestions
    """

    def __init__(self, ram_threshold: float = 0.8,
                 cpu_threshold: float = 0.9,
                 monitoring_interval: float = 1.0):
        self.ram_threshold = ram_threshold
        self.cpu_threshold = cpu_threshold
        self.monitoring_interval = monitoring_interval

        self.history = {
            'cpu_percent': [],
            'ram_percent': [],
            'ram_gb': [],
            'timestamps': []
        }

        self.baseline_measured = False
        self.baseline_cpu = 0.0
        self.baseline_ram = 0.0

    def start_monitoring(self):
        """Start background resource monitoring"""
        import threading

        def monitor_loop():
            while True:
                self._record_snapshot()
                time.sleep(self.monitoring_interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _record_snapshot(self):
        """Record current resource usage"""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        ram_gb = psutil.virtual_memory().used / (1024**3)

        timestamp = time.time()

        self.history['cpu_percent'].append(cpu_percent)
        self.history['ram_percent'].append(ram_percent)
        self.history['ram_gb'].append(ram_gb)
        self.history['timestamps'].append(timestamp)

        # Keep only recent history (last 100 points)
        max_history = 100
        if len(self.history['cpu_percent']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]

        # Set baseline on first measurement
        if not self.baseline_measured:
            self.baseline_cpu = cpu_percent
            self.baseline_ram = ram_percent
            self.baseline_measured = True

    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        if not self.history['cpu_percent']:
            return {'cpu_percent': 0.0, 'ram_percent': 0.0, 'ram_gb': 0.0}

        return {
            'cpu_percent': self.history['cpu_percent'][-1],
            'ram_percent': self.history['ram_percent'][-1],
            'ram_gb': self.history['ram_gb'][-1]
        }

    def should_scale_down(self) -> Tuple[bool, str]:
        """
        Check if we should scale down complexity

        Returns: (should_scale, reason)
        """
        if not self.history['ram_percent']:
            return False, "No data yet"

        current_ram = self.history['ram_percent'][-1]
        current_cpu = self.history['cpu_percent'][-1]

        if current_ram > self.ram_threshold * 100:
            return True, f"RAM usage {current_ram:.1f}% exceeds threshold {self.ram_threshold*100:.1f}%"

        if current_cpu > self.cpu_threshold * 100:
            return True, f"CPU usage {current_cpu:.1f}% exceeds threshold {self.cpu_threshold*100:.1f}%"

        return False, "Resource usage within limits"

    def get_scaling_suggestion(self, current_particle_count: int) -> Tuple[int, str]:
        """
        Suggest optimal particle count based on resource usage

        Returns: (suggested_count, reason)
        """
        current_usage = self.get_current_usage()

        if current_usage['ram_percent'] > 80:
            suggested = max(10, current_particle_count // 2)
            return suggested, f"High RAM usage ({current_usage['ram_percent']:.1f}%), reduce particles by 50%"

        elif current_usage['cpu_percent'] > 70:
            suggested = max(10, current_particle_count * 3 // 4)
            return suggested, f"High CPU usage ({current_usage['cpu_percent']:.1f}%), reduce particles by 25%"

        elif current_usage['ram_percent'] < 50 and current_usage['cpu_percent'] < 50:
            suggested = min(1000, current_particle_count * 3 // 2)
            return suggested, f"Low resource usage, can increase particles by 50%"

        return current_particle_count, "Resource usage optimal, maintain current scale"

    def get_performance_summary(self) -> Dict[str, Union[float, str]]:
        """Get summary of performance metrics"""
        if not self.history['cpu_percent']:
            return {"status": "No data collected yet"}

        cpu_avg = np.mean(self.history['cpu_percent'])
        ram_avg = np.mean(self.history['ram_percent'])
        ram_peak = max(self.history['ram_percent'])

        return {
            "cpu_average": cpu_avg,
            "ram_average": ram_avg,
            "ram_peak": ram_peak,
            "measurements": len(self.history['cpu_percent']),
            "duration_minutes": (self.history['timestamps'][-1] - self.history['timestamps'][0]) / 60
        }

    def print_status(self):
        """Print current resource status"""
        usage = self.get_current_usage()
        should_scale, reason = self.should_scale_down()

        print(f"[ResourceMonitor] CPU: {usage['cpu_percent']:.1f}%, "
              f"RAM: {usage['ram_percent']:.1f}% ({usage['ram_gb']:.1f}GB)")

        if should_scale:
            print(f"⚠️  {reason}")
        else:
            print("✅ Resource usage within limits")


# Integration function for easy use
def create_subatomic_lattice(num_particles: int = 50,
                           particle_types: Optional[List[ParticleType]] = None,
                           max_states: Optional[int] = None) -> Tuple[List[Particle], QuantumStateApproximator, ResourceMonitor]:
    """
    Create a complete subatomic lattice system

    Returns: (particles, approximator, monitor)
    """
    if particle_types is None:
        # Mix of particle types
        particle_types = [ParticleType.QUARK] * (num_particles // 3) + \
                        [ParticleType.LEPTON] * (num_particles // 3) + \
                        [ParticleType.BOSON] * (num_particles - 2 * (num_particles // 3))

    # Create particles with random properties
    particles = []
    for ptype in particle_types:
        # Random quantum properties
        charge = np.random.choice([-1, -1/3, 0, 1/3, 1])
        spin = np.random.choice([0, 1/2, 1, 3/2, 2])
        mass = np.random.uniform(0.1, 10.0)

        particle = Particle(
            particle_type=ptype,
            position=np.random.uniform(-10, 10, 3),
            momentum=np.random.uniform(-5, 5, 3),
            charge=charge,
            spin=spin,
            mass=mass
        )
        particles.append(particle)


    # Create approximator and monitor
    # Use configured max_states, default to num_particles if not specified
    effective_max_states = max_states if max_states is not None else num_particles
    approximator = QuantumStateApproximator(entropy_threshold=0.3, max_states=effective_max_states)
    monitor = ResourceMonitor()

    # Start resource monitoring
    monitor.start_monitoring()

    return particles, approximator, monitor


# Module-level docstring
"""
🎨 SUBATOMIC LATTICE = QUANTUM → GENETIC BRIDGE

This module creates the fundamental building blocks where:
- Quantum particles carry "genetic information" through their properties
- Allelic traits emerge from quantum numbers
- State approximation prevents exponential complexity
- Resource monitoring enables potato-friendly operation

The lattice represents reality's building blocks, where physics and biology meet.
"""

