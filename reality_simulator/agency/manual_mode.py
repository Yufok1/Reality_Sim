"""
👤 MANUAL AGENCY MODE

Human-in-the-loop decision making with comprehensive logging
for training future AI agents.

Features:
- Interactive decision prompts
- Batch processing for efficiency
- Comprehensive decision logging
- Strategy presets
- Performance tracking
"""

import json
import csv
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class DecisionRecord:
    """
    Record of a single human decision
    """
    timestamp: float
    decision_type: str
    context: Dict[str, Any]
    options: List[str]
    chosen_option: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    response_time: float = 0.0
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'decision_type': self.decision_type,
            'context': self.context,
            'options': self.options,
            'chosen_option': self.chosen_option,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'response_time': self.response_time,
            'session_id': self.session_id
        }


@dataclass
class StrategyPreset:
    """
    Predefined decision-making strategies
    """
    name: str
    description: str
    rules: Dict[str, Any]

    def apply(self, context: Dict[str, Any]) -> str:
        """Apply strategy to context and return decision"""
        # This would implement specific strategy logic
        # For now, return a placeholder
        return f"strategy_{self.name}_decision"


class DecisionLogger:
    """
    Logs all human decisions for analysis and AI training
    """

    def __init__(self, log_dir: str = "data/decision_logs"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_log = []

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # File paths
        self.json_log_path = os.path.join(log_dir, f"decisions_{self.session_id}.json")
        self.csv_log_path = os.path.join(log_dir, f"decisions_{self.session_id}.csv")

    def log_decision(self, record: DecisionRecord):
        """Log a decision record"""
        record.session_id = self.session_id
        self.current_session_log.append(record)

        # Immediate write to avoid data loss
        self._write_json_record(record)
        self._write_csv_record(record)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        if not self.current_session_log:
            return {"decisions_made": 0}

        decisions = self.current_session_log
        decision_types = {}
        response_times = []

        for decision in decisions:
            # Count decision types
            dt = decision.decision_type
            decision_types[dt] = decision_types.get(dt, 0) + 1

            # Collect response times
            if decision.response_time > 0:
                response_times.append(decision.response_time)

        return {
            "decisions_made": len(decisions),
            "decision_types": decision_types,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "total_session_time": time.time() - decisions[0].timestamp if decisions else 0,
            "decisions_per_minute": len(decisions) / ((time.time() - decisions[0].timestamp) / 60) if decisions else 0
        }

    def export_training_data(self, output_path: Optional[str] = None) -> str:
        """
        Export decision data in format suitable for AI training

        Returns path to exported file
        """
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"training_data_{self.session_id}.json")

        training_data = []
        for record in self.current_session_log:
            # Convert to training format
            training_example = {
                "context": record.context,
                "options": record.options,
                "chosen_option": record.chosen_option,
                "decision_type": record.decision_type,
                "reasoning": record.reasoning,
                "confidence": record.confidence
            }
            training_data.append(training_example)

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        return output_path

    def _write_json_record(self, record: DecisionRecord):
        """Write single record to JSON log"""
        # For simplicity, we'll append to a list in JSON format
        # In production, consider a more robust logging system
        try:
            # Read existing data
            if os.path.exists(self.json_log_path):
                with open(self.json_log_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []

            # Append new record
            data.append(record.to_dict())

            # Write back
            with open(self.json_log_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not write to JSON log: {e}")

    def _write_csv_record(self, record: DecisionRecord):
        """Write single record to CSV log"""
        try:
            # Prepare CSV row
            row = {
                'timestamp': record.timestamp,
                'decision_type': record.decision_type,
                'chosen_option': record.chosen_option,
                'reasoning': record.reasoning or '',
                'confidence': record.confidence or '',
                'response_time': record.response_time,
                'session_id': record.session_id,
                'context_summary': str(record.context)[:100],  # Truncate for CSV
                'options_count': len(record.options)
            }

            # Check if file exists to write header
            file_exists = os.path.exists(self.csv_log_path)

            with open(self.csv_log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        except Exception as e:
            print(f"Warning: Could not write to CSV log: {e}")


class ManualAgency:
    """
    Human-in-the-loop decision making system

    Handles all interactive decision-making with comprehensive logging
    """

    def __init__(self, logger: Optional[DecisionLogger] = None):
        self.logger = logger or DecisionLogger()
        self.strategy_presets = self._load_strategy_presets()
        self.pending_decisions: List[DecisionRecord] = []

    def _load_strategy_presets(self) -> Dict[str, StrategyPreset]:
        """Load predefined decision-making strategies"""
        return {
            'conservative': StrategyPreset(
                name='conservative',
                description='Prefer safe, established options. Avoid risk.',
                rules={'risk_tolerance': 0.2, 'innovation_factor': 0.1}
            ),
            'balanced': StrategyPreset(
                name='balanced',
                description='Balance risk and reward. Moderate approach.',
                rules={'risk_tolerance': 0.5, 'innovation_factor': 0.5}
            ),
            'innovative': StrategyPreset(
                name='innovative',
                description='Embrace novelty and risk. Favor new possibilities.',
                rules={'risk_tolerance': 0.8, 'innovation_factor': 0.9}
            ),
            'chaos': StrategyPreset(
                name='chaos',
                description='Maximum unpredictability. Random decisions.',
                rules={'random_factor': 1.0}
            )
        }

    def make_decision(self, decision_type: str, context: Dict[str, Any],
                     options: List[str], batch_mode: bool = False) -> str:
        """
        Make a decision with human input

        Args:
            decision_type: Type of decision (e.g., "network_connection")
            context: Decision context information
            options: Available options
            batch_mode: If True, collect decision for later processing

        Returns:
            Chosen option
        """
        if batch_mode:
            # Queue decision for batch processing
            record = DecisionRecord(
                timestamp=time.time(),
                decision_type=decision_type,
                context=context,
                options=options,
                chosen_option="",  # Will be filled later
                session_id=self.logger.session_id
            )
            self.pending_decisions.append(record)
            return "queued"  # Placeholder

        # Immediate decision
        return self._prompt_decision(decision_type, context, options)

    def process_batch_decisions(self) -> List[str]:
        """
        Process all pending batch decisions interactively

        Returns list of chosen options in same order as pending_decisions
        """
        if not self.pending_decisions:
            return []

        print(f"\n{'='*60}")
        print(f"🤔 BATCH DECISIONS ({len(self.pending_decisions)} pending)")
        print(f"{'='*60}")
        print("You can process these one by one, or use a strategy preset.")
        print()

        # Offer strategy presets
        print("Strategy Presets:")
        for name, preset in self.strategy_presets.items():
            print(f"  {name}: {preset.description}")
        print("  manual: Decide each one individually")
        print()

        strategy_choice = input("Choose strategy (conservative/balanced/innovative/chaos/manual): ").strip().lower()

        chosen_options = []

        if strategy_choice in self.strategy_presets:
            # Apply strategy to all decisions
            preset = self.strategy_presets[strategy_choice]
            print(f"\nApplying '{strategy_choice}' strategy to all {len(self.pending_decisions)} decisions...")

            for record in self.pending_decisions:
                chosen_option = preset.apply(record.context)
                record.chosen_option = chosen_option
                record.reasoning = f"Strategy: {strategy_choice}"
                record.confidence = 0.5  # Strategy-based, not certain
                record.response_time = 0.1  # Fast

                self.logger.log_decision(record)
                chosen_options.append(chosen_option)

        else:
            # Manual processing
            print("\nProcessing decisions manually...")
            for i, record in enumerate(self.pending_decisions, 1):
                print(f"\nDecision {i}/{len(self.pending_decisions)}")
                chosen_option = self._prompt_decision(
                    record.decision_type, record.context, record.options
                )

                record.chosen_option = chosen_option
                self.logger.log_decision(record)
                chosen_options.append(chosen_option)

        # Clear pending decisions
        self.pending_decisions.clear()

        print(f"\n✅ Batch processing complete! {len(chosen_options)} decisions logged.")
        return chosen_options

    def _prompt_decision(self, decision_type: str, context: Dict[str, Any],
                        options: List[str]) -> str:
        """
        Prompt user for a single decision
        """
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"🤔 DECISION: {decision_type.upper()}")
        print(f"{'='*50}")

        # Show context
        print("Context:")
        for key, value in context.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, list) and len(value) > 3:
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")
        print()

        # Show options
        print("Options:")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print()

        # Get choice
        while True:
            try:
                choice_input = input(f"Choose option (1-{len(options)}): ").strip()

                # Check for strategy shortcuts
                if choice_input.lower() in ['c', 'con', 'conservative']:
                    choice = self.strategy_presets['conservative'].apply(context)
                    break
                elif choice_input.lower() in ['b', 'bal', 'balanced']:
                    choice = self.strategy_presets['balanced'].apply(context)
                    break
                elif choice_input.lower() in ['i', 'inn', 'innovative']:
                    choice = self.strategy_presets['innovative'].apply(context)
                    break
                elif choice_input.lower() in ['chaos', 'random', 'r']:
                    choice = self.strategy_presets['chaos'].apply(context)
                    break

                # Normal numeric choice
                choice_idx = int(choice_input) - 1
                if 0 <= choice_idx < len(options):
                    choice = options[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(options)}")

            except ValueError:
                print("Please enter a valid number or strategy name")

        response_time = time.time() - start_time

        # Optional reasoning
        reasoning = input("Optional reasoning (press Enter to skip): ").strip()
        if not reasoning:
            reasoning = None

        # Optional confidence
        confidence_input = input("Confidence (0.0-1.0, press Enter for auto): ").strip()
        try:
            confidence = float(confidence_input) if confidence_input else None
        except ValueError:
            confidence = None

        # Log decision
        record = DecisionRecord(
            timestamp=start_time,
            decision_type=decision_type,
            context=context,
            options=options,
            chosen_option=choice,
            reasoning=reasoning,
            confidence=confidence,
            response_time=response_time
        )
        self.logger.log_decision(record)

        return choice

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get comprehensive decision-making statistics"""
        session_stats = self.logger.get_session_stats()

        # Add agency-specific stats
        agency_stats = {
            "pending_decisions": len(self.pending_decisions),
            "strategy_presets_available": len(self.strategy_presets),
            "decision_quality_estimate": self._estimate_decision_quality()
        }

        return {**session_stats, **agency_stats}

    def _estimate_decision_quality(self) -> float:
        """Estimate quality of human decision-making"""
        if not self.logger.current_session_log:
            return 0.0

        decisions = self.logger.current_session_log

        # Quality factors
        avg_confidence = np.mean([d.confidence for d in decisions if d.confidence is not None])
        avg_response_time = np.mean([d.response_time for d in decisions if d.response_time > 0])

        # Normalize (faster responses with higher confidence = higher quality)
        confidence_score = avg_confidence if avg_confidence else 0.5
        speed_score = min(1.0, 10.0 / (avg_response_time + 1.0))  # Faster = better

        return (confidence_score + speed_score) / 2.0

    def export_training_data(self) -> str:
        """Export decision data for AI training"""
        return self.logger.export_training_data()


# Quick utility functions
def create_manual_agency(log_dir: str = "data/decision_logs") -> ManualAgency:
    """Create a manual agency with logging"""
    logger = DecisionLogger(log_dir)
    return ManualAgency(logger)


def get_decision_summary(log_dir: str = "data/decision_logs") -> Dict[str, Any]:
    """Get summary of all decision logs in directory"""
    if not os.path.exists(log_dir):
        return {"error": "Log directory does not exist"}

    summary = {
        "total_sessions": 0,
        "total_decisions": 0,
        "decision_types": {},
        "avg_response_time": 0.0
    }

    try:
        for filename in os.listdir(log_dir):
            if filename.startswith("decisions_") and filename.endswith(".json"):
                filepath = os.path.join(log_dir, filename)
                with open(filepath, 'r') as f:
                    session_data = json.load(f)

                summary["total_sessions"] += 1
                summary["total_decisions"] += len(session_data)

                # Analyze decision types
                for decision in session_data:
                    dt = decision.get("decision_type", "unknown")
                    summary["decision_types"][dt] = summary["decision_types"].get(dt, 0) + 1

                    # Response time
                    rt = decision.get("response_time", 0)
                    if rt > 0:
                        summary["avg_response_time"] += rt

        # Calculate averages
        if summary["total_decisions"] > 0:
            summary["avg_response_time"] /= summary["total_decisions"]

    except Exception as e:
        summary["error"] = str(e)

    return summary


# Module-level docstring
"""
👤 MANUAL AGENCY

The foundation of human-AI symbiosis: Humans make decisions, systems learn.

- Interactive decision prompts with context
- Comprehensive logging for AI training
- Strategy presets for efficiency
- Batch processing for complex scenarios
- Performance tracking and analytics

This creates the training data that future AI agents will learn from.
"""

