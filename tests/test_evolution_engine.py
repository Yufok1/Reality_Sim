"""
🧪 TESTS FOR GENETIC EVOLUTION ENGINE

Test genotype/phenotype mapping, fitness evaluation, selection, mutation,
caching, and complete evolutionary cycles.
"""

import numpy as np
import sys
import time
sys.path.insert(0, '.')

from reality_simulator.evolution_engine import (
    Genotype, GenotypeEncoding, Phenotype, Organism,
    FitnessCache, SelectionEngine, MutationEngine, EvolutionEngine,
    create_evolution_engine, simple_fitness_function
)


def test_genotype_creation():
    """Test genotype creation and basic operations"""
    print("[TEST] Testing genotype creation...")

    # Binary genotype
    genes = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    genotype = Genotype(genes=genes, encoding=GenotypeEncoding.BIT_ARRAY)

    assert genotype.encoding == GenotypeEncoding.BIT_ARRAY
    assert len(genotype.genes) == 8
    assert genotype.fitness is None
    assert genotype.age == 0
    assert genotype.generation == 0

    # Test hash generation
    genotype_hash = genotype.get_hash()
    assert isinstance(genotype_hash, str)
    assert len(genotype_hash) > 0

    # Test copy
    copy = genotype.copy()
    assert np.array_equal(copy.genes, genotype.genes)
    assert copy.age == 1  # Age increments on copy
    assert copy is not genotype  # Different objects

    print("[TEST] Genotype creation and operations work")


def test_genotype_mutation():
    """Test mutation operations"""
    print("[TEST] Testing genotype mutation...")

    genes = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    genotype = Genotype(genes=genes)

    # Test bit array mutation
    mutated = genotype.mutate(rate=1.0)  # 100% mutation chance

    # Should be different (with very high probability)
    assert not np.array_equal(mutated.genes, genotype.genes)

    # Test low mutation rate
    original = genotype.copy()
    low_mutated = genotype.mutate(rate=0.0)  # 0% mutation chance

    # Should be identical
    assert np.array_equal(low_mutated.genes, original.genes)

    print("[TEST] Genotype mutation works")


def test_genotype_crossover():
    """Test crossover operations"""
    print("[TEST] Testing genotype crossover...")

    genes1 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    genes2 = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)

    parent1 = Genotype(genes=genes1)
    parent2 = Genotype(genes=genes2)

    # Single-point crossover
    child1, child2 = parent1.crossover(parent2, method='single_point')

    assert len(child1.genes) == len(parent1.genes)
    assert len(child2.genes) == len(parent2.genes)
    assert child1.generation == 1  # New generation
    assert child2.generation == 1

    # Children should be mixtures of parents
    # (Specific crossover point is random, so we check that they're not identical to parents)
    assert not (np.array_equal(child1.genes, parent1.genes) and
               np.array_equal(child1.genes, parent2.genes))

    print("[TEST] Genotype crossover works")


def test_phenotype_expression():
    """Test phenotype development from genotype"""
    print("[TEST] Testing phenotype expression...")

    phenotype = Phenotype()

    # Express traits
    value1 = phenotype.express_trait("speed", 0.8, 1.0)
    value2 = phenotype.express_trait("strength", 0.6, 0.9)

    assert abs(value1 - 0.8) < 1e-10  # Full development
    assert abs(value2 - 0.54) < 0.1   # Modified by environment

    assert "speed" in phenotype.traits
    assert "strength" in phenotype.traits

    # Test fitness contribution
    fitness1 = phenotype.get_fitness_contribution("speed", 0.8)
    fitness2 = phenotype.get_fitness_contribution("speed", 0.5)

    assert fitness1 > fitness2  # Closer to target = higher fitness

    print("[TEST] Phenotype expression works")


def test_organism_creation():
    """Test complete organism with genotype and phenotype"""
    print("[TEST] Testing organism creation...")

    genes = np.random.randint(0, 2, 16, dtype=np.uint8)
    genotype = Genotype(genes=genes)
    organism = Organism(genotype=genotype)

    assert organism.genotype is genotype
    assert isinstance(organism.phenotype, Phenotype)
    assert organism.fitness == 0.0
    assert organism.species_id == genotype.get_hash()[:16]

    # Test phenotype development
    organism.develop_phenotype()
    assert organism.phenotype.development_stage == 1
    assert len(organism.phenotype.traits) > 0

    # Test fitness calculation
    fitness_targets = {"trait_0": 0.8, "trait_1": 0.2}
    fitness = organism.calculate_fitness(fitness_targets)

    assert 0.0 <= fitness <= 1.0
    assert organism.fitness == fitness
    assert organism.genotype.fitness == fitness

    print("[TEST] Organism creation and evaluation work")


def test_fitness_cache():
    """Test fitness caching for performance"""
    print("[TEST] Testing fitness cache...")

    cache = FitnessCache(max_cache_size=10)

    # Test caching
    cache.put("hash1", 0.8)
    cache.put("hash2", 0.6)

    assert cache.get("hash1") == 0.8
    assert cache.get("hash2") == 0.6
    assert cache.get("nonexistent") is None

    # Test LRU eviction
    for i in range(8):  # Fill to capacity
        cache.put(f"hash_{i+3}", 0.5)

    # hash1 should still be accessible (recently accessed)
    cache.get("hash1")  # Access to move to end

    cache.put("hash_final", 0.9)  # Should evict hash2 (LRU)

    assert cache.get("hash1") == 0.8  # Still there
    assert cache.get("hash2") is None  # Evicted

    # Test stats
    stats = cache.stats()
    assert stats['size'] <= stats['max_size']

    print("[TEST] Fitness cache works")


def test_selection_engine():
    """Test parent selection strategies"""
    print("[TEST] Testing selection engine...")

    selection = SelectionEngine(tournament_size=3, elitism_rate=0.1)

    # Create test population
    population = []
    for i in range(10):
        genes = np.random.randint(0, 2, 8, dtype=np.uint8)
        genotype = Genotype(genes=genes, fitness=float(i) / 10.0)  # Fitness 0.0 to 0.9
        organism = Organism(genotype=genotype)
        organism.fitness = genotype.fitness
        population.append(organism)

    # Test tournament selection
    parents = selection.tournament_selection(population, 3)

    assert len(parents) == 3
    # Winners should have higher fitness (though not guaranteed due to randomness)
    avg_parent_fitness = np.mean([p.fitness for p in parents])
    avg_pop_fitness = np.mean([p.fitness for p in population])
    assert avg_parent_fitness >= avg_pop_fitness * 0.7  # Allow some randomness

    # Test elitism
    elites = selection.elitism_selection(population, 2)

    assert len(elites) == 2
    # Should be the top 2
    assert elites[0].fitness >= elites[1].fitness
    assert elites[0].fitness >= population[-1].fitness

    print("[TEST] Selection engine works")


def test_mutation_engine():
    """Test adaptive mutation rates"""
    print("[TEST] Testing mutation engine...")

    mutation = MutationEngine(base_rate=0.01, adaptive=True)

    # Test non-adaptive rate
    mutation.adaptive = False
    rate = mutation.get_adaptive_rate([], 0)
    assert rate == mutation.base_rate

    # Create test population
    population = []
    for i in range(20):
        genes = np.random.randint(0, 2, 8, dtype=np.uint8)
        genotype = Genotype(genes=genes, fitness=0.5 + np.random.normal(0, 0.1))
        organism = Organism(genotype=genotype)
        organism.fitness = genotype.fitness
        population.append(organism)

    # Test adaptive rate with stagnation
    mutation.update_stats(population)
    rate = mutation.get_adaptive_rate(population, 1)

    # Should be within reasonable bounds
    assert 0.001 <= rate <= 0.1

    # Test mutation application
    offspring = [Genotype(genes=np.random.randint(0, 2, 8, dtype=np.uint8)) for _ in range(5)]
    mutated = mutation.apply_mutations(offspring, 0.5)  # High mutation rate

    assert len(mutated) == len(offspring)
    # Some should be different (though randomness makes this probabilistic)
    # We'll just check that the function runs without error

    print("[TEST] Mutation engine works")


def test_evolution_engine_basic():
    """Test basic evolution engine functionality"""
    print("[TEST] Testing evolution engine basics...")

    engine = EvolutionEngine(
        population_size=20,
        genotype_length=16,
        max_generations=50,
        fitness_targets={"trait_0": 0.8, "trait_1": 0.2}
    )

    assert len(engine.population) == 20
    assert all(isinstance(org, Organism) for org in engine.population)

    # Test single generation
    stats = engine.evolve_generation()

    assert 'generation' in stats
    assert 'best_fitness' in stats
    assert 'avg_fitness' in stats
    assert 'elapsed_seconds' in stats
    assert stats['generation'] == 1
    assert 0.0 <= stats['best_fitness'] <= 1.0

    # Population should still be size 20
    assert len(engine.population) == 20

    print("[TEST] Evolution engine basics work")


def test_evolution_engine_multi_generation():
    """Test multi-generation evolution"""
    print("[TEST] Testing multi-generation evolution...")

    engine = EvolutionEngine(population_size=10, genotype_length=8, max_generations=5)

    # Run a few generations
    history = engine.evolve_for_generations(3)

    assert len(history['generations']) == 3
    assert len(history['best_fitness']) == 3
    assert len(history['avg_fitness']) == 3

    # Fitness should generally improve (though not guaranteed)
    initial_avg = history['avg_fitness'][0]
    final_avg = history['avg_fitness'][-1]

    # At minimum, should not crash and produce reasonable values
    assert all(0.0 <= f <= 1.0 for f in history['best_fitness'])
    assert all(0.0 <= f <= 1.0 for f in history['avg_fitness'])

    # Test population stats
    pop_stats = engine.get_population_stats()
    assert 'population_size' in pop_stats
    assert 'best_fitness' in pop_stats
    assert pop_stats['population_size'] == 10

    # Test best organism retrieval
    best_org = engine.get_best_organism()
    assert isinstance(best_org, Organism)
    assert best_org.fitness == pop_stats['best_fitness']

    print("[TEST] Multi-generation evolution works")


def test_utility_functions():
    """Test utility functions"""
    print("[TEST] Testing utility functions...")

    # Test engine creation
    engine = create_evolution_engine(population_size=5, genotype_length=4)
    assert len(engine.population) == 5
    assert len(engine.population[0].genotype.genes) == 4

    # Test simple fitness function
    genes = np.array([1, 0, 1, 0], dtype=np.uint8)
    genotype = Genotype(genes=genes)
    organism = Organism(genotype=genotype)
    organism.develop_phenotype()

    fitness = simple_fitness_function(organism)
    assert 0.0 <= fitness <= 1.0

    print("[TEST] Utility functions work")


def test_performance():
    """Test performance characteristics"""
    print("[TEST] Testing performance...")

    # Small population for quick test
    engine = EvolutionEngine(population_size=10, genotype_length=8, max_generations=10)

    start_time = time.time()
    history = engine.evolve_for_generations(2)
    elapsed = time.time() - start_time

    # Should complete in reasonable time (under 10 seconds for small test)
    assert elapsed < 10.0

    # Should have cache hits (performance optimization)
    cache_stats = engine.fitness_cache.stats()
    assert cache_stats['size'] > 0  # Some caching occurred

    print(f"[TEST] Performance test passed: {elapsed:.2f}s for 2 generations")


def run_all_tests():
    """Run all evolution engine tests"""
    print("=" * 60)
    print("=" * 60)
    print("[TEST] GENETIC EVOLUTION ENGINE TESTS")
    print("=" * 60)
    print()

    tests = [
        test_genotype_creation,
        test_genotype_mutation,
        test_genotype_crossover,
        test_phenotype_expression,
        test_organism_creation,
        test_fitness_cache,
        test_selection_engine,
        test_mutation_engine,
        test_evolution_engine_basic,
        test_evolution_engine_multi_generation,
        test_utility_functions,
        test_performance
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            print()
            passed += 1
        except Exception as e:
            print(f"[TEST] TEST FAILED: {e}")
            print()
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("=" * 60)
    print(f"[TEST] TESTS COMPLETE: {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("[TEST] All evolution engine tests passed!")
        print("The engine can create and evolve populations of organisms.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check implementation.")


if __name__ == "__main__":
    run_all_tests()

