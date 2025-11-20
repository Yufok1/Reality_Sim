"""
🚦 AGENCY ROUTER

Coordinates between manual and AI decision-making modes.
Routes decisions based on context, user preference, and AI confidence.

Features:
- Automatic mode switching based on AI performance
- User preference override
- Decision quality monitoring
- Seamless fallback between modes
- Performance analytics across modes
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .manual_mode import ManualAgency
from .network_decision_agent import NetworkDecisionAgent


class AgencyMode(Enum):
    """Decision-making modes"""
    MANUAL_ONLY = "manual_only"      # Human makes all decisions
    AI_ASSISTED = "ai_assisted"      # AI suggests, human confirms uncertain ones
    AI_AUTONOMOUS = "ai_autonomous"  # AI makes decisions, human can override
    HYBRID = "hybrid"               # Mix of manual and AI based on decision type


@dataclass
class DecisionRouting:
    """
    Routing configuration for different decision types
    """
    decision_type: str
    preferred_mode: AgencyMode
    ai_confidence_threshold: float = 0.6
    allow_override: bool = True
    batch_capable: bool = False

    def should_use_ai(self, ai_performance: float) -> bool:
        """Determine if AI should be used for this decision type"""
        if self.preferred_mode == AgencyMode.MANUAL_ONLY:
            return False
        elif self.preferred_mode == AgencyMode.AI_AUTONOMOUS:
            return True
        elif self.preferred_mode == AgencyMode.AI_ASSISTED:
            return ai_performance >= self.ai_confidence_threshold
        else:  # HYBRID
            return ai_performance >= 0.5  # Lower threshold for hybrid


@dataclass
class AgencyPerformance:
    """
    Performance metrics for agency system
    """
    total_decisions: int = 0
    manual_decisions: int = 0
    ai_decisions: int = 0
    deferred_decisions: int = 0
    accuracy_vs_manual: float = 0.0
    avg_response_time: float = 0.0
    user_satisfaction: float = 0.0  # Based on override frequency

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "total_decisions": self.total_decisions,
            "ai_adoption_rate": self.ai_decisions / max(1, self.total_decisions),
            "deferred_rate": self.deferred_decisions / max(1, self.total_decisions),
            "manual_rate": self.manual_decisions / max(1, self.total_decisions),
            "avg_response_time": self.avg_response_time,
            "user_satisfaction": self.user_satisfaction
        }


class AgencyRouter:
    """
    Routes decisions between manual and AI modes
    """

    def __init__(self,
                 manual_agency: ManualAgency,
                 ai_agent: Optional[NetworkDecisionAgent] = None,
                 initial_mode: AgencyMode = AgencyMode.MANUAL_ONLY):
        self.manual_agency = manual_agency
        self.ai_agent = ai_agent  # Always None - agents removed
        self.current_mode = AgencyMode.MANUAL_ONLY  # Force manual only

        # Decision routing configuration
        self.routing_config: Dict[str, DecisionRouting] = self._create_default_routing()

        # Performance tracking
        self.performance = AgencyPerformance()

        # Mode transition history
        self.mode_history: List[Tuple[float, AgencyMode, str]] = []

        # Override tracking
        self.pending_overrides: Dict[str, Any] = {}

    def _create_default_routing(self) -> Dict[str, DecisionRouting]:
        """Create default routing configuration for different decision types"""
        return {
            "network_connection": DecisionRouting(
                decision_type="network_connection",
                preferred_mode=AgencyMode.AI_ASSISTED,
                ai_confidence_threshold=0.6,
                allow_override=True,
                batch_capable=True
            ),
            "evolution_mutation": DecisionRouting(
                decision_type="evolution_mutation",
                preferred_mode=AgencyMode.MANUAL_ONLY,  # Critical decisions manual
                ai_confidence_threshold=0.8,
                allow_override=False,
                batch_capable=False
            ),
            "resource_allocation": DecisionRouting(
                decision_type="resource_allocation",
                preferred_mode=AgencyMode.HYBRID,
                ai_confidence_threshold=0.7,
                allow_override=True,
                batch_capable=True
            )
        }

    def make_decision(self, decision_type: str,
                     context: Dict[str, Any],
                     options: List[str],
                     force_mode: Optional[AgencyMode] = None) -> str:
        """
        Route decision to appropriate agency mode

        Args:
            decision_type: Type of decision
            context: Decision context
            options: Available options
            force_mode: Override automatic routing

        Returns:
            Chosen option
        """
        start_time = time.time()
        routing = self.routing_config.get(decision_type)

        if not routing:
            # Default to manual for unknown decision types
            routing = DecisionRouting(decision_type, AgencyMode.MANUAL_ONLY)

        # Determine which mode to use
        effective_mode = force_mode or self.current_mode

        # AI agents removed - always use manual
        ai_performance = 0.0

        # Route decision - always manual (agents removed)
        result = self._route_to_manual(decision_type, context, options, routing)

        response_time = time.time() - start_time

        # Update performance metrics
        self._update_performance_metrics(decision_type, result, response_time, routing)

        return result

    def _route_to_manual(self, decision_type: str, context: Dict[str, Any],
                        options: List[str], routing: DecisionRouting) -> str:
        """Route to manual agency"""
        batch_mode = routing.batch_capable and len(options) > 2

        result = self.manual_agency.make_decision(
            decision_type, context, options, batch_mode=batch_mode
        )

        if result == "queued":
            # Handle batch processing
            batch_results = self.manual_agency.process_batch_decisions()
            result = batch_results[0] if batch_results else options[0]

        self.performance.manual_decisions += 1
        return result

    def _route_to_ai(self, decision_type: str, context: Dict[str, Any],
                    options: List[str], routing: DecisionRouting) -> str:
        """Route to AI agent - disabled, always use manual"""
        return self._route_to_manual(decision_type, context, options, routing)

    def _route_to_assisted(self, decision_type: str, context: Dict[str, Any],
                          options: List[str], routing: DecisionRouting) -> str:
        """AI-assisted mode - disabled, always use manual"""
        return self._route_to_manual(decision_type, context, options, routing)

                    result = ai_suggestion if human_decision == ai_suggestion else \
                            ("no" if ai_suggestion == "yes" else "yes")

                    self.performance.manual_decisions += 1
                    self.performance.deferred_decisions += 1
                    return result

        # Fallback to manual
        return self._route_to_manual(decision_type, context, options, routing)

    def _route_to_hybrid(self, decision_type: str, context: Dict[str, Any],
                        options: List[str], routing: DecisionRouting) -> str:
        """Hybrid mode - disabled, always use manual"""
        return self._route_to_manual(decision_type, context, options, routing)

    def _handle_override(self, decision_type: str, context: Dict[str, Any],
                        options: List[str]) -> str:
        """Handle user override for AI decision"""
        print(f"\n⚠️  AI deferred decision: {decision_type}")
        print("AI was uncertain, please decide:")

        override_decision = self.manual_agency.make_decision(
            f"{decision_type}_override", context, options
        )

        return override_decision

    def _update_performance_metrics(self, decision_type: str, result: str,
                                  response_time: float, routing: DecisionRouting):
        """Update performance tracking"""
        self.performance.total_decisions += 1
        self.performance.avg_response_time = (
            self.performance.avg_response_time * (self.performance.total_decisions - 1) +
            response_time
        ) / self.performance.total_decisions

    def switch_mode(self, new_mode: AgencyMode, reason: str = ""):
        """Switch agency mode"""
        old_mode = self.current_mode
        self.current_mode = new_mode

        self.mode_history.append((time.time(), new_mode, reason))

        print(f"🚦 Agency mode switched: {old_mode.value} → {new_mode.value}")
        if reason:
            print(f"Reason: {reason}")

    def adaptive_mode_switching(self):
        """Automatically switch modes based on performance"""
        # AI agents removed - no performance tracking
        ai_performance = 0.0
        deferred_rate = 0.0

        # Switch logic disabled - always manual only (agents removed)

    def get_status(self) -> Dict[str, Any]:
        """Get current agency status"""
        return {
            "current_mode": self.current_mode.value,
            "manual_agency_stats": self.manual_agency.get_decision_stats(),
            "ai_agent_stats": {},
            "performance_summary": self.performance.get_summary(),
            "routing_config": {k: v.preferred_mode.value for k, v in self.routing_config.items()},
            "mode_history": [(t, m.value, r) for t, m, r in self.mode_history[-5:]]  # Last 5
        }

    def process_deferred_decisions(self):
        """Process deferred decisions - disabled, no agents"""
        pass

    def export_decision_data(self) -> Tuple[str, str]:
        """Export decision data from manual agency only"""
        manual_path = self.manual_agency.export_training_data()
        return manual_path, ""


# Utility functions
def create_agency_router(manual_log_dir: str = "data/decision_logs",
                        ai_model: str = "none",
                        initial_mode: AgencyMode = AgencyMode.MANUAL_ONLY) -> AgencyRouter:
    """Create a complete agency router system (manual only - agents removed)"""
    from .manual_mode import create_manual_agency

    manual_agency = create_manual_agency(manual_log_dir)
    # ai_agent removed - agents completely removed

    return AgencyRouter(manual_agency, ai_agent=None, initial_mode=AgencyMode.MANUAL_ONLY)


def get_agency_recommendation(performance_data: Dict[str, Any]) -> AgencyMode:
    """
    Recommend agency mode based on performance data
    """
    ai_confidence = performance_data.get("ai_confidence", 0.0)
    deferred_rate = performance_data.get("deferred_rate", 1.0)
    total_decisions = performance_data.get("total_decisions", 0)

    if total_decisions < 10:
        return AgencyMode.MANUAL_ONLY  # Not enough data

    if ai_confidence > 0.8 and deferred_rate < 0.1:
        return AgencyMode.AI_AUTONOMOUS
    elif ai_confidence > 0.6 and deferred_rate < 0.3:
        return AgencyMode.AI_ASSISTED
    else:
        return AgencyMode.MANUAL_ONLY


# Module-level docstring
"""
🚦 AGENCY ROUTER

The traffic cop of human-AI decision making:

- Routes decisions based on type, confidence, and user preference
- Automatic mode switching when AI performance changes
- Seamless fallback between manual and AI modes
- Performance monitoring across all decision pathways
- Override handling for critical decisions

Manual mode creates training data for AI improvement.
"""

