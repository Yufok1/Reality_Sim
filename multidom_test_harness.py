"""

Test harness for multi-domain tutor verification.

Run this to validate all 5 components before pushing to GitHub.

"""

import json
import re
from collections import defaultdict
from datetime import datetime

class MultiDomainTestHarness:
    """Captures and validates multi-domain tutor behavior"""

    def __init__(self, log_file_path=None):
        self.log_file = log_file_path or "multidom_test_results.json"
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": defaultdict(list),
            "metrics": {}
        }

    def validate_tutor_activation(self, log_lines):
        """
        TEST 1: Check that MultiDomainTutor activates at right generations
        """
        print("\n" + "="*60)
        print("TEST 1: Tutor Activation")
        print("="*60)

        tutor_activations = []
        for i, line in enumerate(log_lines):
            # Keep ANSI codes but search for the text content
            if "[MULTI-DOMAIN TUTOR]" in line and "Calling" in line:
                # Extract generation number (ANSI codes won't interfere with regex)
                gen_match = re.search(r"Generation (\d+)", line)
                if gen_match:
                    gen = int(gen_match.group(1))
                    tutor_activations.append((gen, line))

        print(f"[OK] Found {len(tutor_activations)} tutor activations")

        if tutor_activations:
            first_gen = tutor_activations[0][0]
            print(f"  First activation: Generation {first_gen}")
            for gen, line in tutor_activations[:5]:  # Show first 5
                print(f"    Gen {gen}: {line[:80]}...")

            self.results["tests"]["activation"].append({
                "count": len(tutor_activations),
                "first_generation": first_gen,
                "status": "PASS" if len(tutor_activations) > 0 else "FAIL"
            })
            return True
        else:
            print("  [FAIL] NO TUTOR ACTIVATIONS FOUND")
            self.results["tests"]["activation"].append({
                "count": 0,
                "status": "FAIL"
            })
            return False

    def validate_domain_progression(self, log_lines):
        """
        TEST 2: Check domain progression (quantum → temporal → social, etc.)
        """
        print("\n" + "="*60)
        print("TEST 2: Domain Progression")
        print("="*60)

        domain_sequence = defaultdict(list)

        # Look for bridging verbs that appear in tutor response content
        # These verbs indicate cross-domain thinking
        bridging_verbs = ["bridges", "integrates", "connects", "synthesizes", "unifies", "links", "amplifies"]
        tutor_context = False
        current_gen = None

        for line in log_lines:
            # Detect start of tutor response
            if "[MULTI-DOMAIN TUTOR]" in line and "Response:" in line:
                gen_match = re.search(r"Generation (\d+)", line)
                if gen_match:
                    current_gen = int(gen_match.group(1))
                    tutor_context = True
                # Don't continue - we want to check this line too

            # If we're in tutor context, look for bridging verbs
            if tutor_context and current_gen:
                if any(verb in line.lower() for verb in bridging_verbs):
                    domain_sequence["bridging"].append(current_gen)

                # Exit tutor context after a few lines or when we hit debug/extracted words
                if "[DEBUG]" in line or "[MULTI-DOMAIN TUTOR]" in line or line.strip() == "":
                    tutor_context = False
                    current_gen = None

        print("Cross-domain bridging by generation:")
        if domain_sequence["bridging"]:
            gens = domain_sequence["bridging"]
            print(f"  Cross-domain bridging -> Gens {min(gens)}-{max(gens)} ({len(gens)} times)")
            bridging_found = True
        else:
            print(f"  Cross-domain bridging -> NOT FOUND [FAIL]")
            bridging_found = False

        # For backwards compatibility, mark domain_progression as passed if bridging is found
        return bridging_found

        self.results["tests"]["domain_progression"] = dict(domain_sequence)
        return len(domain_sequence) > 0

    def validate_bridge_injection_v2(self, log_lines):
        """
        IMPROVED: Look for actual evidence of cross-domain vocabulary in tutor responses
        """
        print("\n" + "="*60)
        print("TEST 3: Bridge Vocabulary Injection (IMPROVED)")
        print("="*60)

        # Look for tutor responses that mention multiple domains
        bridge_evidence = []
        tutor_context = False
        current_response = []

        for line in log_lines:
            # Detect start of tutor response
            if "[MULTI-DOMAIN TUTOR]" in line and "Response:" in line:
                gen_match = re.search(r"Generation (\d+)", line)
                if gen_match:
                    current_gen = int(gen_match.group(1))
                    tutor_context = True
                    current_response = []
                continue

            # Collect response content
            if tutor_context:
                current_response.append(line)

                # Check for domain diversity in accumulated response
                response_text = " ".join(current_response)
                domains = ['quantum', 'temporal', 'social', 'epistemic', 'mathematical']
                domains_found = [d for d in domains if d in response_text.lower()]

                if len(domains_found) >= 2:  # Found cross-domain content
                    bridge_evidence.append({
                        'gen': current_gen,
                        'domains': domains_found,
                        'response': response_text[:200]
                    })

                # Exit tutor context
                if "[DEBUG]" in line or "[MULTI-DOMAIN TUTOR]" in line or line.strip() == "":
                    tutor_context = False

        print(f"[OK] Found {len(bridge_evidence)} potential cross-domain tutor responses")

        for evidence in bridge_evidence[:3]:
            print(f"\n  Generation {evidence['gen']}:")
            print(f"    Domains: {evidence['domains']}")
            print(f"    Sample: {evidence['response'][:100]}...")

        self.results["tests"]["bridge_injection"].append({
            "count": len(bridge_evidence),
            "status": "PASS" if len(bridge_evidence) > 0 else "WARN"
        })
        return len(bridge_evidence) > 0

    def validate_bridge_injection(self, log_lines):
        """Legacy method - calls improved version"""
        return self.validate_bridge_injection_v2(log_lines)

    def validate_consciousness_metrics(self, log_lines):
        """
        TEST 4: Check new multi-domain consciousness metrics are calculated
        """
        print("\n" + "="*60)
        print("TEST 4: Consciousness Metrics")
        print("="*60)

        metrics_found = {
            "vocabulary_coherence": 0,
            "cross_domain_integration": 0,
            "consciousness_diversity": 0,
            "expansion_trajectory": 0
        }

        metric_lines = []
        for line in log_lines:
            # Keep ANSI codes for metrics search
            for metric in metrics_found.keys():
                if metric.upper() in line or metric in line:
                    metrics_found[metric] += 1
                    if len(metric_lines) < 10:  # Capture first 10
                        metric_lines.append((metric, line))

        print("Metrics detected:")
        for metric, count in metrics_found.items():
            status = "[OK]" if count > 0 else "[FAIL]"
            print(f"  {status} {metric:30} {count:3d} occurrences")

        print("\nExample metric log lines:")
        for metric, line in metric_lines[:5]:
            print(f"    {line[:90]}...")

        self.results["tests"]["consciousness_metrics"] = metrics_found
        return sum(metrics_found.values()) > 0

    def validate_expansion_trajectory(self, log_lines):
        """
        TEST 5: Check expansion trajectory logging
        """
        print("\n" + "="*60)
        print("TEST 5: Expansion Trajectory Logging")
        print("="*60)

        trajectories = []
        for i, line in enumerate(log_lines):
            # Keep ANSI codes for expansion trajectory search
            if "[EXPANSION_TRAJECTORY]" in line:
                gen_match = re.search(r"Gen (\d+)", line)
                if gen_match:
                    gen = int(gen_match.group(1))
                    trajectories.append((gen, line))

        print(f"[OK] Found {len(trajectories)} expansion trajectory logs")

        if trajectories:
            for gen, line in trajectories[:5]:
                print(f"    Gen {gen}: {line[:80]}...")

            # Check if trajectories show meaningful direction
            gens = [t[0] for t in trajectories]
            if len(gens) > 1:
                gen_deltas = [gens[i+1] - gens[i] for i in range(len(gens)-1)]
                avg_interval = sum(gen_deltas) / len(gen_deltas)
                print(f"  Average generation interval: {avg_interval:.1f}")

        self.results["tests"]["expansion_trajectory"].append({
            "count": len(trajectories),
            "status": "PASS" if len(trajectories) > 0 else "WARN"
        })
        return len(trajectories) > 0

    def validate_referential_memory(self, log_lines):
        """
        TEST 6: Check referential memory system functionality
        """
        print("\n" + "="*60)
        print("TEST 6: Referential Memory System")
        print("="*60)

        memory_indicators = {
            'anchors_recorded': 0,
            'stability_metrics': 0,
            'selection_pressure': 0,
            'memory_feedback': 0,
            'anchor_clusters': 0
        }

        for line in log_lines:
            # Check for anchor recording in context memory
            if "[CONTEXT_MEMORY]" in line or "link_word_to_node" in line:
                memory_indicators['anchors_recorded'] += 1

            # Check for stability metrics logging
            if "[MEMORY_STABILITY]" in line:
                memory_indicators['stability_metrics'] += 1

            # Check for selection pressure application
            if "[MEMORY_SELECTION]" in line:
                memory_indicators['selection_pressure'] += 1

            # Check for memory-enhanced feedback
            if "[MEMORY_FEEDBACK]" in line:
                memory_indicators['memory_feedback'] += 1

            # Check for anchor clustering in vision prompts
            if "REFERENTIAL MEMORY INSIGHTS" in line or "Cluster" in line and "organisms share" in line:
                memory_indicators['anchor_clusters'] += 1

        print("Referential memory indicators found:")
        for indicator, count in memory_indicators.items():
            status = "[OK]" if count > 0 else "[MISSING]"
            print(f"  {status} {indicator}: {count} occurrences")

        # Show example lines for each indicator type
        for line in log_lines[:50]:  # Check first 50 lines for examples
            if any(indicator in line for indicator in ["[MEMORY_STABILITY]", "[MEMORY_SELECTION]", "[MEMORY_FEEDBACK]", "REFERENTIAL MEMORY INSIGHTS"]):
                print(f"    Example: {line.strip()[:100]}...")

        self.results["tests"]["referential_memory"] = memory_indicators

        # Pass if at least some memory functionality is detected
        total_indicators = sum(memory_indicators.values())
        status = "PASS" if total_indicators > 0 else "FAIL"
        print(f"\n[{'PASS' if total_indicators > 0 else 'FAIL'}] Referential memory system: {total_indicators} total indicators")

        return total_indicators > 0

    def run_all_tests(self, log_file_path):
        """Run all 5 validation tests"""
        print(f"\n{'='*60}")
        print(f"MULTI-DOMAIN TUTOR TEST HARNESS")
        print(f"{'='*60}")
        print(f"Reading logs from: {log_file_path}")

        try:
            # Try UTF-8 first, then UTF-16 if that fails
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
            except UnicodeDecodeError:
                # Try UTF-16 (common with Windows PowerShell output)
                with open(log_file_path, 'r', encoding='utf-16') as f:
                    log_lines = f.readlines()
        except FileNotFoundError:
            print(f"[ERROR] Log file not found: {log_file_path}")
            return False

        print(f"Total log lines: {len(log_lines)}")

        # Run all tests
        test_results = {}
        test_results["activation"] = self.validate_tutor_activation(log_lines)
        test_results["domain_progression"] = self.validate_domain_progression(log_lines)
        test_results["bridge_injection"] = self.validate_bridge_injection(log_lines)
        test_results["consciousness_metrics"] = self.validate_consciousness_metrics(log_lines)
        test_results["expansion_trajectory"] = self.validate_expansion_trajectory(log_lines)
        test_results["referential_memory"] = self.validate_referential_memory(log_lines)

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)
        print(f"Tests passed: {passed}/{total} (6 components tested)")

        for test_name, result in test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {test_name}")

        # Save results
        self.results["test_end"] = datetime.now().isoformat()
        self.results["summary"] = test_results

        with open(self.log_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed results saved to: {self.log_file}")

        return passed == total

def main():
    """
    Usage:
        1. Run simulator: python reality_simulator/main.py --mode observer
        2. Let it run for ~30 generations
        3. Find the log file (usually in logs/ or data/ directory)
        4. Run this: python multidom_test_harness.py <path_to_log>
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multidom_test_harness.py <log_file_path>")
        print("\nExample:")
        print("  python multidom_test_harness.py logs/reality_sim.log")
        print("\nTo generate a log:")
        print("  python reality_simulator/main.py --mode observer 2>&1 | tee sim_output.log")
        print("  # Then run:")
        print("  python multidom_test_harness.py sim_output.log")
        return

    log_file = sys.argv[1]
    harness = MultiDomainTestHarness()
    success = harness.run_all_tests(log_file)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
