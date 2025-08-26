
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Callable, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from loguru import logger

try:
    from .modular_system import ModularSystem
except ImportError:
    from modular_system import ModularSystem


class AlgorithmResult(Enum):
    """Possible results of a reconfiguration algorithm."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    INVALID = "invalid"


@dataclass
class TestResult:
    """Results from running a test scenario."""
    scenario_name: str
    algorithm_name: str
    initial_connectivity: bool
    final_connectivity: bool
    result: AlgorithmResult
    execution_time: float
    steps_taken: int
    connectivity_restored_at_step: int
    initial_components: int
    final_components: int
    initial_metrics: Dict[str, Any]
    final_metrics: Dict[str, Any]
    pivot_operations: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""


@dataclass
class TestScenario:
    """Defines a test scenario for the reconfiguration problem."""
    name: str
    description: str
    system_generator: Callable[[], ModularSystem]
    damage_pattern: List[str]  # Module IDs to damage
    expected_result: AlgorithmResult
    timeout_seconds: float = 60.0
    validate_lattice: bool = True


class ReconfigurationAlgorithm(Protocol):
    """Protocol defining the interface for reconfiguration algorithms."""
    
    def __call__(self, system: ModularSystem, max_steps: int = 100) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Execute the reconfiguration algorithm.
        
        Args:
            system: The modular system to reconfigure
            max_steps: Maximum number of pivot operations allowed
            
        Returns:
            Tuple of (success, pivot_operations, error_message)
        """
        ...


class BaseReconfigurationAlgorithm(ABC):
    """Base class for reconfiguration algorithms."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, system: ModularSystem, max_steps: int = 100) -> Tuple[bool, List[Dict[str, Any]], str]:
        """Execute the reconfiguration algorithm."""
        pass
    
    def __call__(self, system: ModularSystem, max_steps: int = 100) -> Tuple[bool, List[Dict[str, Any]], str]:
        return self.execute(system, max_steps)


class DummyAlgorithm(BaseReconfigurationAlgorithm):
    """
    Dummy algorithm for testing the framework.
    """
    
    def __init__(self):
        super().__init__("Dummy")
    
    def execute(self, system: ModularSystem, max_steps: int = 100) -> Tuple[bool, List[Dict[str, Any]], str]:
        """Execute dummy algorithm that just reports system state."""
        pivot_operations = []
        
        # Check if already connected
        if system.is_connected():
            return True, pivot_operations, "Success: System already connected"
        
        # Get some basic info about the system
        components = system.get_connected_components()
        pivotable = system.get_pivotable_modules()
        
        # Create a dummy operation
        if pivotable:
            operation = {
                'step': 0,
                'pivot_module': pivotable[0],
                'components_found': len(components),
                'type': 'dummy_operation'
            }
            pivot_operations.append(operation)
        
        return False, pivot_operations, "Dummy algorithm - no actual reconfiguration performed"


class TestBench:
    """
    Comprehensive test bench for evaluating reconfiguration algorithms.
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.algorithms: Dict[str, ReconfigurationAlgorithm] = {}
        self.scenarios: List[TestScenario] = []
        self.results: List[TestResult] = []
        
        # Register default algorithm
        self.register_algorithm("dummy", DummyAlgorithm())
        
        logger.info("TestBench initialized")
    
    def register_algorithm(self, name: str, algorithm: ReconfigurationAlgorithm):
        """Register a new algorithm for testing."""
        self.algorithms[name] = algorithm
        logger.info(f"Registered algorithm: {name}")
    
    def add_scenario(self, scenario: TestScenario):
        """Add a test scenario."""
        self.scenarios.append(scenario)
        logger.info(f"Added scenario: {scenario.name}")
    
    def create_simple_grid_scenario(self, size: int = 3, damage_center: bool = True) -> TestScenario:
        """Create a simple grid scenario with optional center damage."""
        def generator():
            system = ModularSystem()
            
            # Create grid
            modules = {}
            for x in range(size):
                for y in range(size):
                    module_id = f"M{x}{y}"
                    position = np.array([x, y, 0], dtype=float)
                    modules[module_id] = system.add_module(module_id, position)
            
            # Connect adjacent modules
            for x in range(size):
                for y in range(size):
                    current_id = f"M{x}{y}"
                    
                    # Connect to right neighbor
                    if x < size - 1:
                        neighbor_id = f"M{x+1}{y}"
                        system.connect_modules(current_id, neighbor_id)
                    
                    # Connect to top neighbor
                    if y < size - 1:
                        neighbor_id = f"M{x}{y+1}"
                        system.connect_modules(current_id, neighbor_id)
            
            return system
        
        damage_pattern = [f"M{size//2}{size//2}"] if damage_center else [f"M0{0}"]
        
        return TestScenario(
            name=f"Grid{size}x{size}_{'Center' if damage_center else 'Corner'}Damage",
            description=f"{size}x{size} grid with {'center' if damage_center else 'corner'} module damage",
            system_generator=generator,
            damage_pattern=damage_pattern,
            expected_result=AlgorithmResult.SUCCESS
        )
    
    def create_line_scenario(self, length: int = 5, damage_index: int = 2) -> TestScenario:
        """Create a linear chain scenario with damage at specified index."""
        def generator():
            system = ModularSystem()
            
            # Create linear chain
            modules = []
            for i in range(length):
                module_id = f"L{i}"
                position = np.array([i, 0, 0], dtype=float)
                modules.append(system.add_module(module_id, position))
            
            # Connect adjacent modules
            for i in range(length - 1):
                system.connect_modules(f"L{i}", f"L{i+1}")
            
            return system
        
        return TestScenario(
            name=f"Line{length}_Damage{damage_index}",
            description=f"Linear chain of {length} modules with damage at position {damage_index}",
            system_generator=generator,
            damage_pattern=[f"L{damage_index}"],
            expected_result=AlgorithmResult.SUCCESS if damage_index not in [0, length-1] else AlgorithmResult.SUCCESS
        )
    
    def create_t_shape_scenario(self, arm_length: int = 3, damage_center: bool = True) -> TestScenario:
        """Create a T-shaped scenario compatible with cubic lattice constraints."""
        def generator():
            system = ModularSystem()
            
            # Create T-shape: horizontal arm + vertical stem
            # Horizontal arm: (-arm_length, 0, 0) to (arm_length, 0, 0)
            for x in range(-arm_length, arm_length + 1):
                module_id = f"H{x+arm_length}"  # H0, H1, H2, H3, H4, H5, H6 for arm_length=3
                position = np.array([x, 0, 0], dtype=float)
                system.add_module(module_id, position)
            
            # Vertical stem: (0, 1, 0) to (0, arm_length, 0)
            for y in range(1, arm_length + 1):
                module_id = f"V{y}"  # V1, V2, V3 for arm_length=3
                position = np.array([0, y, 0], dtype=float)
                system.add_module(module_id, position)
            
            # Connect horizontal arm
            for x in range(-arm_length, arm_length):
                current_id = f"H{x+arm_length}"
                next_id = f"H{x+arm_length+1}"
                system.connect_modules(current_id, next_id)
            
            # Connect vertical stem to center of horizontal arm
            center_id = f"H{arm_length}"  # Center of horizontal arm
            system.connect_modules(center_id, "V1")
            
            # Connect vertical stem
            for y in range(1, arm_length):
                current_id = f"V{y}"
                next_id = f"V{y+1}"
                system.connect_modules(current_id, next_id)
            
            return system
        
        # Damage pattern: center (junction) or end of arm
        if damage_center:
            damage_pattern = [f"H{arm_length}"]  # Center junction
            scenario_name = f"TShape{arm_length}_CenterDamage"
        else:
            damage_pattern = [f"H{arm_length*2}"]  # End of horizontal arm
            scenario_name = f"TShape{arm_length}_EndDamage"
        
        return TestScenario(
            name=scenario_name,
            description=f"T-shaped structure with arm length {arm_length}, {'center' if damage_center else 'end'} damage",
            system_generator=generator,
            damage_pattern=damage_pattern,
            expected_result=AlgorithmResult.SUCCESS if not damage_center else AlgorithmResult.FAILURE
        )
    
    def run_single_test(self, scenario: TestScenario, algorithm_name: str) -> TestResult:
        """Run a single test with specified scenario and algorithm."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not registered")
        
        algorithm = self.algorithms[algorithm_name]
        
        # Generate system
        system = scenario.system_generator()
        initial_metrics = system.get_system_metrics()
        initial_connectivity = system.is_connected()
        initial_components = system.number_connected_components()
        
        # Apply damage
        damage_success = system.simulate_damage(scenario.damage_pattern)
        if not damage_success:
            logger.warning(f"Damage pattern failed for scenario {scenario.name}")
        
        logger.info(f"Running {algorithm_name} on {scenario.name}")
        
        # Run algorithm
        start_time = time.time()
        
        try:
            success, pivot_operations, error_message = algorithm(system, max_steps=100)
            execution_time = time.time() - start_time
            
            if execution_time > scenario.timeout_seconds:
                result = AlgorithmResult.TIMEOUT
            elif success:
                result = AlgorithmResult.SUCCESS
            else:
                result = AlgorithmResult.FAILURE
                
        except Exception as e:
            execution_time = time.time() - start_time
            result = AlgorithmResult.INVALID
            error_message = str(e)
            pivot_operations = []
            success = False
            logger.error(f"Algorithm {algorithm_name} failed: {e}")
        
        # Get final metrics
        final_metrics = system.get_system_metrics()
        final_connectivity = system.is_connected()
        final_components = system.number_connected_components()
        
        # Find when connectivity was restored
        connectivity_restored_at = -1
        for i, op in enumerate(pivot_operations):
            if 'connectivity_check' in op and op['connectivity_check']:
                connectivity_restored_at = i
                break
        
        # Validate lattice constraints if required
        if scenario.validate_lattice:
            valid, violations = system.verify_lattice_constraints()
            if not valid:
                logger.warning(f"Lattice constraints violated: {violations}")
        
        return TestResult(
            scenario_name=scenario.name,
            algorithm_name=algorithm_name,
            initial_connectivity=initial_connectivity,
            final_connectivity=final_connectivity,
            result=result,
            execution_time=execution_time,
            steps_taken=len(pivot_operations),
            connectivity_restored_at_step=connectivity_restored_at,
            initial_components=initial_components,
            final_components=final_components,
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            pivot_operations=pivot_operations,
            error_message=error_message
        )
    
    def run_benchmark_suite(self, algorithms: List[str] = None, scenarios: List[str] = None) -> List[TestResult]:
        """Run a comprehensive benchmark suite."""
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        if scenarios is None:
            # Create default scenarios if none provided
            if not self.scenarios:
                self.add_scenario(self.create_simple_grid_scenario(3, True))
                self.add_scenario(self.create_simple_grid_scenario(4, True))
                self.add_scenario(self.create_line_scenario(5, 2))
                self.add_scenario(self.create_t_shape_scenario(5, 1))
                self.add_scenario(self.create_t_shape_scenario(7, 2))
            
            scenario_names = [s.name for s in self.scenarios]
        else:
            scenario_names = scenarios
        
        results = []
        total_tests = len(algorithms) * len(scenario_names)
        test_count = 0
        
        logger.info(f"Running benchmark suite: {len(algorithms)} algorithms × {len(scenario_names)} scenarios = {total_tests} tests")
        
        for algorithm_name in algorithms:
            for scenario in self.scenarios:
                if scenario.name in scenario_names:
                    test_count += 1
                    logger.info(f"Test {test_count}/{total_tests}: {algorithm_name} on {scenario.name}")
                    
                    try:
                        result = self.run_single_test(scenario, algorithm_name)
                        results.append(result)
                        
                        logger.info(f"Result: {result.result.value} in {result.execution_time:.3f}s with {result.steps_taken} steps")
                        
                    except Exception as e:
                        logger.error(f"Test failed: {e}")
                        # Create failed result
                        failed_result = TestResult(
                            scenario_name=scenario.name,
                            algorithm_name=algorithm_name,
                            initial_connectivity=False,
                            final_connectivity=False,
                            result=AlgorithmResult.INVALID,
                            execution_time=0.0,
                            steps_taken=0,
                            connectivity_restored_at_step=-1,
                            initial_components=0,
                            final_components=0,
                            initial_metrics={},
                            final_metrics={},
                            pivot_operations=[],
                            error_message=str(e)
                        )
                        results.append(failed_result)
        
        self.results.extend(results)
        return results
    
    def generate_report(self, results: List[TestResult] = None) -> Dict[str, Any]:
        """Generate a comprehensive report of test results."""
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Group results by algorithm
        by_algorithm = {}
        for result in results:
            if result.algorithm_name not in by_algorithm:
                by_algorithm[result.algorithm_name] = []
            by_algorithm[result.algorithm_name].append(result)
        
        # Compute statistics
        report = {
            "summary": {
                "total_tests": len(results),
                "algorithms_tested": len(by_algorithm),
                "scenarios_tested": len(set(r.scenario_name for r in results))
            },
            "algorithm_performance": {},
            "scenario_analysis": {},
            "detailed_results": []
        }
        
        # Algorithm performance analysis
        for alg_name, alg_results in by_algorithm.items():
            success_count = sum(1 for r in alg_results if r.result == AlgorithmResult.SUCCESS)
            avg_time = np.mean([r.execution_time for r in alg_results])
            avg_steps = np.mean([r.steps_taken for r in alg_results])
            
            report["algorithm_performance"][alg_name] = {
                "success_rate": success_count / len(alg_results),
                "average_execution_time": avg_time,
                "average_steps": avg_steps,
                "total_tests": len(alg_results)
            }
        
        # Scenario analysis
        by_scenario = {}
        for result in results:
            if result.scenario_name not in by_scenario:
                by_scenario[result.scenario_name] = []
            by_scenario[result.scenario_name].append(result)
        
        for scenario_name, scenario_results in by_scenario.items():
            success_count = sum(1 for r in scenario_results if r.result == AlgorithmResult.SUCCESS)
            
            report["scenario_analysis"][scenario_name] = {
                "success_rate": success_count / len(scenario_results),
                "algorithms_tested": len(scenario_results),
                "difficulty_score": 1.0 - (success_count / len(scenario_results))
            }
        
        # Add detailed results
        for result in results:
            report["detailed_results"].append({
                "scenario": result.scenario_name,
                "algorithm": result.algorithm_name,
                "result": result.result.value,
                "execution_time": result.execution_time,
                "steps_taken": result.steps_taken,
                "connectivity_restored": result.final_connectivity,
                "error": result.error_message
            })
        
        return report
    
    def save_report(self, report: Dict[str, Any] = None, filename: str = None) -> Path:
        """Save the test report to a JSON file."""
        if report is None:
            report = self.generate_report()
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def print_summary(self, results: List[TestResult] = None):
        """Print a summary of test results to the console."""
        if results is None:
            results = self.results
        
        if not results:
            print("No test results available")
            return
        
        report = self.generate_report(results)
        
        print("\n" + "="*60)
        print("DAMAGE-RESPONSIVE RECONFIGURATION TEST RESULTS")
        print("="*60)
        
        print(f"\nSummary:")
        print(f"  Total tests: {report['summary']['total_tests']}")
        print(f"  Algorithms: {report['summary']['algorithms_tested']}")
        print(f"  Scenarios: {report['summary']['scenarios_tested']}")
        
        print(f"\nAlgorithm Performance:")
        for alg, perf in report['algorithm_performance'].items():
            print(f"  {alg}:")
            print(f"    Success rate: {perf['success_rate']:.1%}")
            print(f"    Avg time: {perf['average_execution_time']:.3f}s")
            print(f"    Avg steps: {perf['average_steps']:.1f}")
        
        print(f"\nScenario Difficulty:")
        for scenario, analysis in report['scenario_analysis'].items():
            print(f"  {scenario}: {analysis['success_rate']:.1%} success rate")
        
        print("="*60)


def main():
    """Main function demonstrating the test bench."""
    logger.info("Starting Test Bench Demo")
    
    # Create test bench
    bench = TestBench()
    
    # Add scenarios
    bench.add_scenario(bench.create_simple_grid_scenario(3, True))
    bench.add_scenario(bench.create_simple_grid_scenario(4, False))
    bench.add_scenario(bench.create_line_scenario(5, 2))
    bench.add_scenario(bench.create_t_shape_scenario(6, 1))
    
    # Run benchmark
    results = bench.run_benchmark_suite()
    
    # Print results
    bench.print_summary(results)
    
    # Save report
    bench.save_report()
    
    logger.info("Test Bench Demo completed")


if __name__ == "__main__":
    main()