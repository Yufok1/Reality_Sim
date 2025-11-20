"""
🤖 NETWORK DECISION AGENT (MINIMAL STUB - NO AI)

This file contains minimal stubs for backwards compatibility.
No AI/Ollama functionality - simulation runs on pure evolution/network/quantum physics.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class NetworkDecisionContext:
    """Minimal context stub"""
    organism_a_id: str
    organism_b_id: str
    organism_a_fitness: float
    organism_b_fitness: float
    organism_a_traits: Dict[str, float]
    organism_b_traits: Dict[str, float]
    current_network_size: int
    connection_history: List[Tuple[str, str]]
    ecosystem_stability: float
    
    def to_prompt_text(self) -> str:
        """Stub for test compatibility"""
        return f"Network Connection Decision Context:\n\nOrganism A (ID: {self.organism_a_id}):\n- Fitness: {self.organism_a_fitness:.3f}\n\nOrganism B (ID: {self.organism_b_id}):\n- Fitness: {self.organism_b_fitness:.3f}\n\nNetwork State:\n- Current size: {self.current_network_size} organisms\n- Ecosystem stability: {self.ecosystem_stability:.2f}"


@dataclass
class DecisionResult:
    """Minimal result stub"""
    decision: str
    confidence: float
    reasoning: Optional[str] = None
    processing_time: float = 0.0
    model_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "processing_time": self.processing_time,
            "model_used": self.model_used
        }


class UncertaintyHandler:
    """Minimal stub"""
    def __init__(self, confidence_threshold: float = 0.6, max_deferred_decisions: int = 10):
        self.confidence_threshold = confidence_threshold
        self.deferred_decisions = []
        self.stats = {
            "total_decisions": 0,
            "deferred_count": 0,
            "avg_confidence": 0.0,
            "uncertainty_rate": 0.0
        }
    
    def should_defer(self, result: DecisionResult) -> bool:
        return result.confidence < self.confidence_threshold
    
    def defer_decision(self, context):  
        self.deferred_decisions.append(context)
    
    def update_stats(self, result: DecisionResult):
        self.stats["total_decisions"] += 1
        if self.should_defer(result):
            self.stats["deferred_count"] += 1
        # Update avg confidence
        total = self.stats["total_decisions"]
        self.stats["avg_confidence"] = (self.stats["avg_confidence"] * (total - 1) + result.confidence) / total
        # Update uncertainty rate
        self.stats["uncertainty_rate"] = self.stats["deferred_count"] / total
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


class OllamaBridge:
    """Minimal stub - No Ollama required"""
    def __init__(self, model_name: str = "none", timeout: float = 30.0, max_retries: int = 3):
        self.model_name = "none"
        self.timeout = timeout
        self.available = False
    
    def make_decision(self, context: NetworkDecisionContext) -> DecisionResult:
        return DecisionResult(decision="uncertain", confidence=0.0, reasoning="No AI - pure physics simulation")
    
    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        return "No AI agent - simulation runs on pure evolution/network/quantum physics"


class NetworkDecisionAgent:
    """
    MINIMAL STUB - No AI functionality
    Simulation runs on pure evolution, network dynamics, and quantum physics
    """
    
    def __init__(self, model_preference: str = "none", model_name: str = "none", 
                 confidence_threshold: float = 0.6):
        self.model_preference = "none"
        self.model_name = "none"
        self.confidence_threshold = confidence_threshold
        self.uncertainty_handler = UncertaintyHandler(confidence_threshold)
        self.decision_history = []
        self.performance_stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "deferred_decisions": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0
        }
    
    def make_decision(self, context: NetworkDecisionContext, network=None) -> DecisionResult:
        """Always returns 'no' - no AI involved"""
        return DecisionResult(
            decision="no",
            confidence=1.0,
            reasoning="No AI - pure physics simulation",
            processing_time=0.0,
            model_used="none"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return empty stats"""
        return self.performance_stats.copy()
    
    def decide_connection(self, org_a_id: str, org_b_id: str, network=None) -> Tuple[bool, DecisionResult]:
        """Stub - always returns (False, DecisionResult)"""
        result = DecisionResult(decision="no", confidence=1.0, reasoning="No AI - pure physics")
        # Update stats for test compatibility
        self.performance_stats["total_decisions"] += 1
        self.performance_stats["successful_decisions"] += 1
        return False, result
    
    def process_user_message(self, user_message: str, simulation_data: Dict[str, Any]) -> str:
        """Minimal chat stub - just returns status"""
        if 'evolution' in simulation_data:
            evo = simulation_data['evolution']
            return f"Gen {evo.get('generation', 0)}: Pop {evo.get('population_size', 0)}, Fitness {evo.get('best_fitness', 0):.3f}"
        return "Simulation running (no AI - pure physics)"


# Utility functions
def create_network_decision_agent(model: str = "none", confidence_threshold: float = 0.6) -> NetworkDecisionAgent:
    """Create stub agent"""
    return NetworkDecisionAgent(
        model_preference="none",
        model_name="none",
        confidence_threshold=confidence_threshold
    )
