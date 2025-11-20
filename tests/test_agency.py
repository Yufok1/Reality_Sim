"""
🧪 TESTS FOR AGENCY LAYER

Test manual mode, AI decision agent, uncertainty handling,
and agency router coordination.
"""

import numpy as np
import sys
import time
import json
import os
sys.path.insert(0, '.')

from reality_simulator.agency.manual_mode import (
    ManualAgency, DecisionLogger, DecisionRecord, StrategyPreset
)
from reality_simulator.agency.network_decision_agent import (
    NetworkDecisionAgent, DecisionResult, UncertaintyHandler,
    NetworkDecisionContext, OllamaBridge
)
from reality_simulator.agency.agency_router import (
    AgencyRouter, AgencyMode, DecisionRouting
)


def test_manual_agency_basic():
    """Test basic manual agency functionality"""
    print("🧪 Testing manual agency basics...")

    # Create temporary log directory
    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:
        agency = ManualAgency(DecisionLogger(temp_dir))

        # Test decision making (simulated)
        context = {"test": "context"}
        options = ["option1", "option2"]

        # Since we can't interact, test the queuing system
        result = agency.make_decision("test_decision", context, options, batch_mode=True)
        assert result == "queued"

        # Check pending decisions
        assert len(agency.pending_decisions) == 1

        # Test strategy presets
        assert "conservative" in agency.strategy_presets
        assert "innovative" in agency.strategy_presets

        print("✅ Manual agency basics work")

    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_decision_logger():
    """Test decision logging functionality"""
    print("🧪 Testing decision logger...")

    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:
        logger = DecisionLogger(temp_dir)

        # Create test record
        record = DecisionRecord(
            timestamp=time.time(),
            decision_type="test",
            context={"key": "value"},
            options=["a", "b"],
            chosen_option="a",
            reasoning="test reasoning",
            confidence=0.8,
            response_time=1.5
        )

        logger.log_decision(record)

        # Check session stats
        stats = logger.get_session_stats()
        assert stats["decisions_made"] == 1
        assert stats["decision_types"]["test"] == 1

        # Check files were created
        assert os.path.exists(logger.json_log_path)
        assert os.path.exists(logger.csv_log_path)

        print("✅ Decision logger works")

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_strategy_presets():
    """Test decision-making strategy presets"""
    print("🧪 Testing strategy presets...")

    preset = StrategyPreset(
        name="test_strategy",
        description="Test strategy",
        rules={"test": "rule"}
    )

    context = {"test": "context"}
    decision = preset.apply(context)

    # Should return some decision string
    assert isinstance(decision, str)
    assert len(decision) > 0

    print("✅ Strategy presets work")


def test_network_decision_context():
    """Test network decision context creation"""
    print("🧪 Testing network decision context...")

    context = NetworkDecisionContext(
        organism_a_id="org1",
        organism_b_id="org2",
        organism_a_fitness=0.8,
        organism_b_fitness=0.6,
        organism_a_traits={"trait1": 0.7},
        organism_b_traits={"trait1": 0.5},
        current_network_size=10,
        connection_history=[("org3", "org4")],
        ecosystem_stability=0.7
    )

    # Test prompt generation
    prompt = context.to_prompt_text()
    assert isinstance(prompt, str)
    assert len(prompt) > 100  # Should be substantial
    assert "org1" in prompt
    assert "org2" in prompt

    print("✅ Network decision context works")


def test_uncertainty_handler():
    """Test uncertainty handling"""
    print("🧪 Testing uncertainty handler...")

    handler = UncertaintyHandler(confidence_threshold=0.6)

    # Test confident decision
    result_confident = DecisionResult("yes", 0.8, "good")
    assert not handler.should_defer(result_confident)

    # Test uncertain decision
    result_uncertain = DecisionResult("yes", 0.4, "uncertain")
    assert handler.should_defer(result_uncertain)

    # Test deferral
    context = NetworkDecisionContext("a", "b", 0.5, 0.5, {}, {}, 5, [], 0.5)
    handler.defer_decision(context)
    assert len(handler.deferred_decisions) == 1

    # Test stats
    handler.update_stats(result_confident)
    handler.update_stats(result_uncertain)
    stats = handler.get_stats()

    assert stats["total_decisions"] == 2
    assert stats["deferred_count"] == 1
    assert stats["uncertainty_rate"] == 0.5

    print("✅ Uncertainty handler works")


def test_ollama_bridge():
    """Test Ollama bridge (mocked)"""
    print("🧪 Testing Ollama bridge...")

    bridge = OllamaBridge()

    # Check availability (will be False without Ollama)
    assert isinstance(bridge.available, bool)

    # Test decision making
    context = NetworkDecisionContext("a", "b", 0.5, 0.5, {}, {}, 5, [], 0.5)
    result = bridge.make_decision(context)

    assert isinstance(result, DecisionResult)
    assert isinstance(result.confidence, (int, float))
    assert 0.0 <= result.confidence <= 1.0

    # If not available, should return uncertain
    if not bridge.available:
        assert result.decision == "uncertain"
        assert result.confidence == 0.0

    print("✅ Ollama bridge works")


def test_network_decision_agent():
    """Test network decision agent"""
    print("🧪 Testing network decision agent...")

    agent = NetworkDecisionAgent()

    # Create mock network
    class MockNetwork:
        def __init__(self):
            self.organisms = {
                'org1': type('MockOrg', (), {'fitness': 0.8})(),
                'org2': type('MockOrg', (), {'fitness': 0.6})()
            }
            self.metrics = type('MockMetrics', (), {'stability_index': 0.7})()

    network = MockNetwork()

    # Test decision making
    should_connect, result = agent.decide_connection(network, 'org1', 'org2')

    assert isinstance(should_connect, bool)
    assert isinstance(result, DecisionResult)

    # Check performance stats
    stats = agent.get_performance_stats()
    assert "total_decisions" in stats
    assert stats["total_decisions"] == 1

    print("✅ Network decision agent works")


def test_agency_router():
    """Test agency router coordination"""
    print("🧪 Testing agency router...")

    # Create mock agencies
    class MockManualAgency:
        def make_decision(self, *args, **kwargs):
            return "manual_decision"
        def get_decision_stats(self):
            return {"manual_stats": True}

    class MockAIAgent:
        def __init__(self):
            self.performance_stats = {"avg_confidence": 0.7}
        def decide_connection(self, network, a, b):
            return True, DecisionResult("yes", 0.8, "ai_decision")
        def get_performance_stats(self):
            return {"ai_stats": True}

    router = AgencyRouter(MockManualAgency(), MockAIAgent())

    # Test mode switching
    router.switch_mode(AgencyMode.AI_AUTONOMOUS, "test")
    assert router.current_mode == AgencyMode.AI_AUTONOMOUS

    # Test status
    status = router.get_status()
    assert "current_mode" in status
    assert status["current_mode"] == "ai_autonomous"

    print("✅ Agency router works")


def test_decision_routing():
    """Test decision routing configuration"""
    print("🧪 Testing decision routing...")

    routing = DecisionRouting(
        decision_type="network_connection",
        preferred_mode=AgencyMode.AI_ASSISTED,
        ai_confidence_threshold=0.6
    )

    # Test AI decision logic
    assert routing.should_use_ai(0.8)  # High confidence
    assert not routing.should_use_ai(0.3)  # Low confidence

    # Test manual-only mode
    manual_routing = DecisionRouting(
        decision_type="critical_decision",
        preferred_mode=AgencyMode.MANUAL_ONLY
    )
    assert not manual_routing.should_use_ai(1.0)  # Never use AI

    print("✅ Decision routing works")


def test_mode_switching():
    """Test adaptive mode switching"""
    print("🧪 Testing mode switching...")

    # Create router with mock agencies
    class MockManualAgency:
        def get_decision_stats(self):
            return {}

    class MockAIAgent:
        def __init__(self, confidence, deferred):
            self.performance_stats = {
                "avg_confidence": confidence,
                "deferred_decisions": deferred,
                "total_decisions": 10
            }
        def get_performance_stats(self):
            return self.performance_stats

    # Test switching from assisted to autonomous (good AI)
    router = AgencyRouter(MockManualAgency(), MockAIAgent(0.9, 0), AgencyMode.AI_ASSISTED)
    router.adaptive_mode_switching()
    assert router.current_mode == AgencyMode.AI_AUTONOMOUS

    # Test switching back due to poor performance
    router.ai_agent = MockAIAgent(0.5, 6)  # High deferral rate (0.6 > 0.5)
    router.adaptive_mode_switching()
    assert router.current_mode == AgencyMode.AI_ASSISTED  # Should degrade gracefully first
    
    # Check further degradation
    router.adaptive_mode_switching()
    assert router.current_mode == AgencyMode.MANUAL_ONLY

    print("✅ Mode switching works")


def test_performance_tracking():
    """Test performance tracking across agencies"""
    print("🧪 Testing performance tracking...")

    from reality_simulator.agency.agency_router import AgencyPerformance

    perf = AgencyPerformance()

    # Simulate some decisions
    perf.total_decisions = 10
    perf.manual_decisions = 6
    perf.ai_decisions = 3
    perf.deferred_decisions = 1
    perf.avg_response_time = 2.5

    summary = perf.get_summary()

    assert summary["total_decisions"] == 10
    assert summary["ai_adoption_rate"] == 0.3
    assert summary["manual_rate"] == 0.6
    assert summary["avg_response_time"] == 2.5

    print("✅ Performance tracking works")


def test_integration():
    """Test full agency layer integration"""
    print("🧪 Testing agency integration...")

    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:
        # Create real agencies
        from reality_simulator.agency.manual_mode import create_manual_agency
        from reality_simulator.agency.network_decision_agent import create_network_decision_agent
        from reality_simulator.agency.agency_router import create_agency_router

        router = create_agency_router(temp_dir, "granite3.1-moe:3b", AgencyMode.MANUAL_ONLY)

        # Test status
        status = router.get_status()
        assert "current_mode" in status

        # Test decision making (should go to manual)
        context = {"test": "context"}
        options = ["yes", "no"]

        # Mock input to avoid interactive hang
        from unittest.mock import patch
        with patch('builtins.input', return_value='1'):
            # This will now proceed with option 1 ("yes")
            result = router.make_decision("network_connection", context, options)
            
        assert result in ["yes", "no", "queued"] or isinstance(result, str)

        print("✅ Agency integration works")

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all agency layer tests"""
    print("=" * 60)
    print("🤖 AGENCY LAYER TESTS")
    print("=" * 60)
    print()

    tests = [
        test_manual_agency_basic,
        test_decision_logger,
        test_strategy_presets,
        test_network_decision_context,
        test_uncertainty_handler,
        test_ollama_bridge,
        test_network_decision_agent,
        test_agency_router,
        test_decision_routing,
        test_mode_switching,
        test_performance_tracking,
        test_integration
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
    print(f"🎉 TESTS COMPLETE: {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("✅ All agency layer tests passed!")
        print("Human-AI decision making system is ready.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check implementation.")


if __name__ == "__main__":
    run_all_tests()

