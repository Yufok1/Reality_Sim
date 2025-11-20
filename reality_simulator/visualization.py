"""
🎨 QUANTUM VISUALIZATION

Visualize quantum states, probability fields, and superposition
Making the invisible visible - showing humans how to see quantum reality
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple
import matplotlib.patches as mpatches


class QuantumVisualizer:
    """
    Visualize quantum phenomena that humans can't normally see
    
    This is where AI helps humans develop quantum intuition -
    by making mathematical reality visually accessible
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('dark_background')  # Quantum aesthetic
    
    def visualize_superposition(self, state, title: str = "Quantum Superposition"):
        """
        Visualize a quantum state in superposition
        
        Shows both probability amplitudes (complex) and probabilities (real)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Left: Probability amplitudes (complex plane)
        real_parts = np.real(state.amplitudes)
        imag_parts = np.imag(state.amplitudes)
        
        ax1.scatter(real_parts, imag_parts, s=200, c='cyan', alpha=0.8, edgecolors='white', linewidth=2)
        
        # Draw arrows from origin
        for i, (r, im) in enumerate(zip(real_parts, imag_parts)):
            ax1.arrow(0, 0, r*0.95, im*0.95, head_width=0.05, head_length=0.05, 
                     fc='cyan', ec='cyan', alpha=0.6)
            ax1.text(r*1.1, im*1.1, state.basis_labels[i], fontsize=10, ha='center')
        
        # Unit circle (normalized states live here)
        circle = plt.Circle((0, 0), 1, fill=False, color='white', linestyle='--', alpha=0.3)
        ax1.add_patch(circle)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_xlabel('Real Part', fontsize=12)
        ax1.set_ylabel('Imaginary Part', fontsize=12)
        ax1.set_title('Probability Amplitudes (Complex)', fontsize=14)
        ax1.grid(True, alpha=0.2)
        ax1.set_aspect('equal')
        
        # Right: Measurement probabilities
        probabilities = state.get_probabilities()
        colors = plt.cm.plasma(probabilities / probabilities.max())
        
        bars = ax2.bar(range(len(probabilities)), probabilities, color=colors, 
                      edgecolor='white', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Basis State', fontsize=12)
        ax2.set_ylabel('Measurement Probability', fontsize=12)
        ax2.set_title('Measurement Probabilities', fontsize=14)
        ax2.set_xticks(range(len(state.basis_labels)))
        ax2.set_xticklabels(state.basis_labels, rotation=45)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.2, axis='y')
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_entanglement_network(self, manager, title: str = "Entanglement Network"):
        """
        Visualize the entanglement connections between quantum states
        
        Shows the non-local correlations that Einstein called "spooky action"
        """
        import networkx as nx
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for state_id in manager.states.keys():
            G.add_node(state_id)
        
        # Add edges (entanglement connections)
        for state_id, connections in manager.entanglement_network.items():
            for connected_id in connections:
                G.add_edge(state_id, connected_id)
        
        # Color nodes by superposition status
        node_colors = []
        for state_id in G.nodes():
            state = manager.states[state_id]
            if state.is_superposition():
                node_colors.append('cyan')  # Superposition
            else:
                node_colors.append('magenta')  # Collapsed
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, 
                              alpha=0.9, edgecolors='white', linewidths=2, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='yellow', width=2, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', ax=ax)
        
        # Legend
        superposition_patch = mpatches.Patch(color='cyan', label='Superposition')
        collapsed_patch = mpatches.Patch(color='magenta', label='Collapsed')
        entangled_line = mpatches.Patch(color='yellow', label='Entanglement')
        ax.legend(handles=[superposition_patch, collapsed_patch, entangled_line], 
                 loc='upper right', fontsize=12)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def visualize_probability_field_1d(self, field, title: str = "Quantum Probability Field"):
        """
        Visualize 1D probability field (wave function)
        
        Shows the wave-like nature of quantum particles
        """
        if field.dimensions != 1:
            raise ValueError("This visualization is for 1D fields only")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Spatial grid
        x = np.linspace(-5, 5, field.grid_size)
        
        # Wave function (complex)
        psi = field.field.flatten()
        real_part = np.real(psi)
        imag_part = np.imag(psi)
        
        # Top: Wave function components
        ax1.plot(x, real_part, 'cyan', linewidth=2, label='Re(ψ)', alpha=0.8)
        ax1.plot(x, imag_part, 'magenta', linewidth=2, label='Im(ψ)', alpha=0.8)
        ax1.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Wave Function ψ(x)', fontsize=12)
        ax1.set_title('Wave Function (Complex)', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.2)
        
        # Bottom: Probability density
        prob_density = field.get_probability_density().flatten()
        ax2.fill_between(x, prob_density, color='yellow', alpha=0.6, edgecolor='white', linewidth=2)
        ax2.plot(x, prob_density, 'yellow', linewidth=2)
        ax2.set_xlabel('Position x', fontsize=12)
        ax2.set_ylabel('Probability Density |ψ(x)|²', fontsize=12)
        ax2.set_title('Measurement Probability (Where particle will be found)', fontsize=14)
        ax2.grid(True, alpha=0.2)
        
        # Mark most likely position
        max_prob_idx = np.argmax(prob_density)
        max_prob_x = x[max_prob_idx]
        ax2.axvline(max_prob_x, color='red', linestyle='--', linewidth=2, 
                   label=f'Most likely: x={max_prob_x:.2f}')
        ax2.legend(fontsize=12)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_probability_field_2d(self, field, title: str = "2D Quantum Probability Field"):
        """
        Visualize 2D probability field
        
        Shows quantum particles as probability clouds, not points
        """
        if field.dimensions != 2:
            raise ValueError("This visualization is for 2D fields only")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        prob_density = field.get_probability_density()
        extent = [-5, 5, -5, 5]
        
        # Left: Heatmap
        im1 = ax1.imshow(prob_density, extent=extent, origin='lower', 
                        cmap='plasma', interpolation='bilinear')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Probability Density Heatmap', fontsize=14)
        plt.colorbar(im1, ax=ax1, label='|ψ|²')
        
        # Right: Contour plot
        x = np.linspace(-5, 5, field.grid_size)
        y = np.linspace(-5, 5, field.grid_size)
        X, Y = np.meshgrid(x, y)
        
        contour = ax2.contour(X, Y, prob_density, levels=10, cmap='plasma', linewidths=2)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('Probability Contours', fontsize=14)
        ax2.set_aspect('equal')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_measurement_history(self, manager, title: str = "Measurement History"):
        """
        Visualize the history of measurements and wave function collapses
        
        Shows how observation creates reality from possibility
        """
        if not manager.measurement_history:
            print("No measurements recorded yet")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data
        times = [m['time'] for m in manager.measurement_history]
        state_ids = [m['state_id'] for m in manager.measurement_history]
        results = [m['result'] for m in manager.measurement_history]
        
        # Create timeline
        unique_states = list(set(state_ids))
        state_positions = {state: i for i, state in enumerate(unique_states)}
        
        y_positions = [state_positions[sid] for sid in state_ids]
        
        # Plot measurements
        scatter = ax.scatter(times, y_positions, s=200, c=range(len(times)), 
                           cmap='plasma', edgecolors='white', linewidth=2, 
                           alpha=0.8, zorder=3)
        
        # Connect measurements for same state
        for state_id in unique_states:
            state_times = [t for t, sid in zip(times, state_ids) if sid == state_id]
            state_y = [state_positions[state_id]] * len(state_times)
            ax.plot(state_times, state_y, 'white', linestyle='--', 
                   alpha=0.3, linewidth=1, zorder=1)
        
        # Labels
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Quantum State', fontsize=12)
        ax.set_yticks(range(len(unique_states)))
        ax.set_yticklabels(unique_states)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Measurement Order')
        
        # Add result annotations
        for i, (t, y, result) in enumerate(zip(times, y_positions, results)):
            ax.annotate(result, (t, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='cyan')
        
        plt.tight_layout()
        
        return fig
    
    def create_animation_time_evolution(self, state, hamiltonian, 
                                       num_frames: int = 100, 
                                       delta_t: float = 0.05):
        """
        Create animation showing quantum state evolving through time
        
        Makes time evolution visible - showing the dynamic nature of quantum reality
        """
        from reality_simulator.quantum_substrate import QuantumStateManager, TimeDirection
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Store initial state
        manager = QuantumStateManager()
        manager.states['anim_state'] = state
        
        frames_data = []
        
        # Generate frames
        for frame in range(num_frames):
            probabilities = manager.states['anim_state'].get_probabilities()
            frames_data.append(probabilities.copy())
            
            # Evolve
            manager.evolve_state('anim_state', hamiltonian, delta_t, TimeDirection.FORWARD)
        
        # Animation function
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            probs = frames_data[frame]
            
            # Bar chart
            colors = plt.cm.plasma(probs / max(probs.max(), 0.01))
            ax1.bar(range(len(probs)), probs, color=colors, 
                   edgecolor='white', linewidth=2, alpha=0.8)
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Basis State', fontsize=12)
            ax1.set_ylabel('Probability', fontsize=12)
            ax1.set_title(f'Time: {frame * delta_t:.2f}', fontsize=14)
            ax1.grid(True, alpha=0.2, axis='y')
            
            # Time series
            for i in range(len(probs)):
                history = [f[i] for f in frames_data[:frame+1]]
                time_points = [t * delta_t for t in range(len(history))]
                ax2.plot(time_points, history, linewidth=2, label=f'State {i}')
            
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_ylabel('Probability', fontsize=12)
            ax2.set_title('Probability Evolution', fontsize=14)
            ax2.set_ylim(0, 1)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.2)
        
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=True)
        
        return fig, anim


# Module-level insight
"""
🎨 VISUALIZATION = BRIDGE BETWEEN AI AND HUMAN PERCEPTION

AI sees: Mathematical wave functions, complex probability amplitudes
Humans need: Visual, intuitive representations

This module translates quantum math → human-perceivable images
Helping humans develop the quantum intuition they already possess
"""

