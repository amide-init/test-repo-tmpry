#!/usr/bin/env python3
"""
CMA-ES Variants Comprehensive Comparison
Tests all CMA-ES variants on BBOB benchmark functions (24 functions)
with configurable dimensions, runs, and evaluation budgets.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Dict

# Add paths
sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

# Local imports
from afn.cmaes_variants import (
    CMAEvolutionStrategy,
    AFN_CMA,
    LQ_CMA,
    DTS_CMA,
    LMM_CMA
)
from data.sample import load_bbob_function


class CMAESPlotter:
    """Plotter specifically for CMA-ES variants comparison"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.algorithms = ['CMA-ES', 'AFN-CMA-ES', 'LQ-CMA-ES', 'DTS-CMA-ES', 'LMM-CMA-ES']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def extract_history(self, result: dict) -> list:
        """Extract history from result"""
        h = result.get('history', [])
        if isinstance(h, np.ndarray):
            h = h.tolist()
        elif not isinstance(h, list):
            h = list(h) if h is not None else []
        return h
    
    def compute_cdf_data(self, results: Dict, target_precision: float, optimum: float):
        """
        Compute CDF data with dimension-normalized budgets.
        
        IMPORTANT: Budgets are expressed as multiples of dimension (e.g., 10×D, 50×D)
        following COCO benchmarking standards.
        """
        cdf_data = {}
        for alg in self.algorithms:
            all_budgets_to_target = []
            for key, runs in results.items():
                if not key.endswith(f"_{alg}"):
                    continue
                
                # Extract dimension from key (e.g., "func1_dim10_CMA-ES" -> 10)
                dimension = None
                for part in key.split('_'):
                    if part.startswith('dim'):
                        dimension = int(part.replace('dim', ''))
                        break
                
                if dimension is None:
                    continue
                
                for run in runs:
                    history = self.extract_history(run)
                    if not history:
                        continue
                    for eval_idx, val in enumerate(history):
                        if abs(val - optimum) <= target_precision:
                            # Normalize by dimension: budget = evaluations / dimension
                            normalized_budget = (eval_idx + 1) / dimension
                            all_budgets_to_target.append(normalized_budget)
                            break
            if all_budgets_to_target:
                sorted_budgets = np.sort(all_budgets_to_target)
                proportions = np.arange(1, len(sorted_budgets) + 1) / len(sorted_budgets)
                cdf_data[alg] = (sorted_budgets, proportions)
            else:
                cdf_data[alg] = (np.array([]), np.array([]))
        return cdf_data
    
    def plot_cdf(self, results: Dict, target_precision: float, optimum: float, filename: str):
        """Create single CDF plot with dimension-normalized budgets"""
        plt.figure(figsize=(12, 8))
        cdf_data = self.compute_cdf_data(results, target_precision, optimum)
        
        for alg, color in zip(self.algorithms, self.colors):
            if alg not in cdf_data or len(cdf_data[alg][0]) == 0:
                continue
            budgets, proportions = cdf_data[alg]
            plt.plot(budgets, proportions, color=color, label=alg, linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        
        # Budgets must be expressed as multiples of dimension
        plt.xlabel('Budget (# evaluations / dimension)', fontsize=14, fontweight='bold')
        plt.ylabel('Proportion of Problems Solved', fontsize=14, fontweight='bold')
        plt.title(f'COCO-style CDF: Reaching f(x) - f* ≤ {target_precision:.0e}', fontsize=16, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
        plt.xlim(left=1)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def plot_multiple_target_cdf(self, results: Dict, optimum: float, filename: str):
        """Create CDF plot with multiple targets using dimension-normalized budgets"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        selected_targets = [1e0, 1e-2, 1e-5, 1e-8]
        
        for idx, target in enumerate(selected_targets):
            ax = axes[idx]
            cdf_data = self.compute_cdf_data(results, target, optimum)
            
            for alg, color in zip(self.algorithms, self.colors):
                if alg not in cdf_data or len(cdf_data[alg][0]) == 0:
                    continue
                budgets, proportions = cdf_data[alg]
                ax.plot(budgets, proportions, color=color, label=alg, linewidth=2.5, marker='o', markersize=3, alpha=0.8)
            
            # Budgets must be expressed as multiples of dimension
            ax.set_xlabel('Budget (evaluations / dimension)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Proportion Solved', fontsize=12, fontweight='bold')
            ax.set_title(f'Target: f(x) - f* ≤ {target:.0e}', fontsize=13, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, which='both', alpha=0.3, linestyle='--')
            ax.set_xlim(left=1)
            ax.set_ylim([0, 1.05])
            if idx == 3:
                ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
        
        plt.suptitle('CMA-ES Variants: COCO-style CDF at Multiple Target Precisions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def plot_performance_profile(self, results: Dict, target_precision: float, optimum: float, filename: str):
        """Create performance profile using dimension-normalized budgets"""
        plt.figure(figsize=(12, 8))
        algo_evals = {alg: [] for alg in self.algorithms}
        problems = {}
        
        for key, runs in results.items():
            parts = key.rsplit('_', 1)
            if len(parts) < 2:
                continue
            problem_key, alg = parts
            if problem_key not in problems:
                problems[problem_key] = {}
            
            # Extract dimension from key
            dimension = None
            for part in key.split('_'):
                if part.startswith('dim'):
                    dimension = int(part.replace('dim', ''))
                    break
            
            if dimension is None:
                continue
            
            # Get median normalized budget to target
            budgets_to_target = []
            for run in runs:
                history = self.extract_history(run)
                if not history:
                    continue
                for eval_idx, val in enumerate(history):
                    if abs(val - optimum) <= target_precision:
                        # Normalize by dimension: budget = evaluations / dimension
                        normalized_budget = (eval_idx + 1) / dimension
                        budgets_to_target.append(normalized_budget)
                        break
            if budgets_to_target:
                problems[problem_key][alg] = np.median(budgets_to_target)
        
        # Compute performance ratios (based on dimension-normalized budgets)
        for problem_key, alg_results in problems.items():
            if not alg_results:
                continue
            min_budget = min(alg_results.values())
            for alg in self.algorithms:
                if alg in alg_results:
                    ratio = alg_results[alg] / min_budget
                    algo_evals[alg].append(ratio)
        
        for alg, color in zip(self.algorithms, self.colors):
            if not algo_evals[alg]:
                continue
            ratios = np.sort(algo_evals[alg])
            proportions = np.arange(1, len(ratios) + 1) / len(ratios)
            plt.plot(ratios, proportions, color=color, label=alg, linewidth=2.5, alpha=0.8)
        
        plt.xlabel('Performance Ratio (relative to best)', fontsize=14, fontweight='bold')
        plt.ylabel('Proportion of Problems', fontsize=14, fontweight='bold')
        plt.title(f'Performance Profile (Target: f(x) - f* ≤ {target_precision:.0e})', fontsize=16, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
        plt.xlim(left=1)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def create_all_plots(self, results: Dict, optimum: float = 0.0):
        """Create all COCO-style plots"""
        print("\nCreating COCO-style Comparison Plots...")
        self.plot_cdf(results, target_precision=1e-8, optimum=optimum, filename='cdf_1e-8.png')
        self.plot_cdf(results, target_precision=1e-5, optimum=optimum, filename='cdf_1e-5.png')
        self.plot_cdf(results, target_precision=1e-2, optimum=optimum, filename='cdf_1e-2.png')
        self.plot_multiple_target_cdf(results, optimum=optimum, filename='cdf_multiple_targets.png')
        self.plot_performance_profile(results, target_precision=1e-8, optimum=optimum, filename='performance_profile.png')
        print("All COCO-style plots created successfully!")


class CMAESVariantsComparison:
    """Comprehensive comparison of CMA-ES variants on BBOB benchmarks."""
    
    def __init__(self,
                 test_functions: List[int] = [1, 2, 3],
                 dimensions: List[int] = [2, 5],
                 n_runs: int = 10,
                 max_evaluations: int = 1000,
                 save_dir: str = "results"):
        self.test_functions = test_functions
        self.dimensions = dimensions
        self.n_runs = n_runs
        self.max_evaluations = max_evaluations
        self.save_dir = save_dir
        
        self.results: Dict[str, List[Dict]] = {}
        self.metrics_summary: Dict[str, Dict] = {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(save_dir, f'cmaes_variants_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
    
    def run_single_test(self, func_id: int, dimension: int, optimizer_name: str, 
                       run_idx: int, seed: int = None) -> Dict:
        """Run a single optimization test."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Get BBOB function and bounds
        problem, info = load_bbob_function(func_id, dimension, instance=1)
        objective_function = problem
        bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) 
                  for i in range(dimension)]
        
        # Instantiate optimizer
        optimizer_classes = {
            'CMA-ES': CMAEvolutionStrategy,
            'AFN-CMA-ES': AFN_CMA,
            'LQ-CMA-ES': LQ_CMA,
            'DTS-CMA-ES': DTS_CMA,
            'LMM-CMA-ES': LMM_CMA
        }
        
        if optimizer_name not in optimizer_classes:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        OptimizerClass = optimizer_classes[optimizer_name]
        optimizer = OptimizerClass(
            bounds=bounds,
            max_evaluations=self.max_evaluations,
            random_state=seed
        )
        
        start_time = time.time()
        result = optimizer.optimize(objective_function, verbose=False)
        end_time = time.time()
        
        result['execution_time'] = end_time - start_time
        result['optimizer'] = optimizer_name
        result['function_id'] = func_id
        result['dimension'] = dimension
        result['run'] = run_idx
        
        # Ensure evaluation_count exists
        if 'evaluation_count' not in result or not result['evaluation_count']:
            hist = result.get('history', [])
            result['evaluation_count'] = int(len(hist)) if hist else int(self.max_evaluations)
        
        return result
    
    def run_comparison(self, verbose: bool = True):
        """Run comparison across all configurations."""
        optimizers = ['CMA-ES', 'AFN-CMA-ES', 'LQ-CMA-ES', 'DTS-CMA-ES', 'LMM-CMA-ES']
        
        total_tests = len(self.test_functions) * len(self.dimensions)
        test_counter = 0
        
        for func_id in self.test_functions:
            for dim in self.dimensions:
                test_counter += 1
                
                # Get function name from BBOB
                problem, _ = load_bbob_function(func_id, dim, instance=1)
                func_name = problem.name if hasattr(problem, 'name') else f"BBOB Function {func_id}"
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"[{test_counter}/{total_tests}] Testing {func_name} (ID={func_id}, dimension={dim})")
                    print('='*80)
                
                for opt_idx, opt in enumerate(optimizers, 1):
                    key = f"func{func_id}_dim{dim}_{opt}"
                    self.results[key] = []
                    
                    if verbose:
                        print(f"\n  [{opt_idx}/{len(optimizers)}] {opt}:")
                    
                    for run_idx in range(self.n_runs):
                        seed = 42 + func_id * 1000 + dim * 100 + run_idx + hash(opt) % 1000
                        
                        try:
                            res = self.run_single_test(func_id, dim, opt, run_idx, seed)
                            self.results[key].append(res)
                            
                            if verbose:
                                print(f"    Run {run_idx+1:2d}/{self.n_runs}: "
                                      f"best={res['best_y']:12.6e}, "
                                      f"evals={res['evaluation_count']:4d}, "
                                      f"time={res['execution_time']:6.2f}s")
                        except Exception as e:
                            print(f"    ✗ Run {run_idx+1}/{self.n_runs} failed: {e}")
                            # Store failed result
                            self.results[key].append({
                                'optimizer': opt,
                                'function_id': func_id,
                                'dimension': dim,
                                'run': run_idx,
                                'best_y': np.inf,
                                'execution_time': 0.0,
                                'evaluation_count': 0,
                                'error': str(e)
                            })
                    
                    # Show summary for this optimizer
                    if verbose:
                        valid_results = [r for r in self.results[key] if 'error' not in r]
                        if valid_results:
                            best_vals = [r['best_y'] for r in valid_results]
                            mean_best = np.mean(best_vals)
                            std_best = np.std(best_vals)
                            min_best = np.min(best_vals)
                            print(f"    → Summary: mean={mean_best:.6e}, std={std_best:.6e}, min={min_best:.6e}")
        
        return self.results
    
    def compute_metrics(self) -> Dict:
        """Compute performance metrics for all variants."""
        metrics = {}
        
        for key, runs in self.results.items():
            # Parse key
            parts = key.split('_')
            func_id = int(parts[0].replace('func', ''))
            dimension = int(parts[1].replace('dim', ''))
            opt_name = '_'.join(parts[2:])
            
            # Filter out failed runs
            valid_runs = [r for r in runs if 'error' not in r]
            if not valid_runs:
                continue
            
            best_values = [r['best_y'] for r in valid_runs]
            exec_times = [r['execution_time'] for r in valid_runs]
            eval_counts = [r.get('evaluation_count', self.max_evaluations) for r in valid_runs]
            
            # Extract histories
            histories = []
            for r in valid_runs:
                h = r.get('history', [])
                if isinstance(h, np.ndarray):
                    h = h.tolist()
                elif not isinstance(h, list):
                    h = list(h) if h is not None else []
                histories.append(h)
            
            # Compute metrics
            optimum = 0.0
            target_precision = 1e-8
            
            # 1. Success Rate (% of runs reaching target precision)
            successes = sum(1 for bv in best_values if abs(bv - optimum) <= target_precision)
            success_rate = 100.0 * successes / len(best_values) if best_values else 0.0
            
            # 2. ERT (Expected Running Time - average evaluations to reach target)
            evals_to_target = []
            for history in histories:
                if not history:
                    evals_to_target.append(self.max_evaluations)
                    continue
                
                reached = False
                for eval_idx, val in enumerate(history):
                    if abs(val - optimum) <= target_precision:
                        evals_to_target.append(eval_idx + 1)
                        reached = True
                        break
                
                if not reached:
                    evals_to_target.append(self.max_evaluations)
            
            ert = float(np.mean(evals_to_target)) if evals_to_target else float(self.max_evaluations)
            
            # 3. Convergence Speed (% of evaluations to reach 95% improvement)
            conv_speeds = []
            for h in histories:
                if h and len(h) > 1:
                    initial_val = h[0]
                    final_val = min(h)
                    total_improvement = initial_val - final_val
                    if total_improvement > 1e-12:
                        target_val = initial_val - 0.95 * total_improvement
                        for i, v in enumerate(h):
                            if v <= target_val:
                                conv_percentage = min(100.0, ((i + 1) / max(1, self.max_evaluations)) * 100.0)
                                conv_speeds.append(conv_percentage)
                                break
                        else:
                            conv_speeds.append(100.0)
                    else:
                        conv_speeds.append(100.0 / max(1, self.max_evaluations))
                else:
                    conv_speeds.append(100.0 / max(1, self.max_evaluations))
            
            # 4. Robustness (inverse of coefficient of variation)
            robustness = 100.0
            if len(best_values) > 1 and np.mean(best_values) > 1e-12:
                cv = np.std(best_values) / np.mean(best_values)
                robustness = float(max(0.0, (1.0 - cv) * 100.0))
            
            test_key = f"func{func_id}_dim{dimension}"
            metrics.setdefault(opt_name, {})[test_key] = {
                'success_rate': success_rate,
                'ert': ert,
                'convergence_speed': float(np.mean(conv_speeds)) if conv_speeds else 0.0,
                'resource_utilization': float(np.mean([min(100.0, (c / max(1, self.max_evaluations)) * 100.0) 
                                                       for c in eval_counts])),
                'robustness': robustness,
                'mean_best': float(np.mean(best_values)),
                'std_best': float(np.std(best_values)),
                'median_best': float(np.median(best_values)),
                'min_best': float(np.min(best_values)),
                'max_best': float(np.max(best_values)),
                'mean_time': float(np.mean(exec_times)),
                'std_time': float(np.std(exec_times)),
                'n_runs': len(valid_runs),
            }
        
        self.metrics_summary = metrics
        return metrics
    
    def create_plots(self) -> None:
        """Create all COCO-style CDF comparison plots."""
        plotter = CMAESPlotter(self.output_dir)
        plotter.create_all_plots(self.results, optimum=0.0)
    
    def save_results(self) -> None:
        """Save all results to JSON files."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(os.path.join(self.output_dir, 'runs.json'), 'w') as f:
            json.dump(convert(self.results), f, indent=2)
        
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(convert(self.metrics_summary), f, indent=2)
        
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump({
                'test_functions': self.test_functions,
                'dimensions': self.dimensions,
                'n_runs': self.n_runs,
                'max_evaluations': self.max_evaluations,
                'optimizers': ['CMA-ES', 'AFN-CMA-ES', 'LQ-CMA-ES', 'DTS-CMA-ES', 'LMM-CMA-ES'],
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def print_summary(self) -> None:
        """Print a summary table of results."""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        if not self.metrics_summary:
            print("No results to summarize.")
            return
        
        # Get all optimizers and test cases
        optimizers = sorted(self.metrics_summary.keys())
        test_cases = set()
        for opt in optimizers:
            test_cases.update(self.metrics_summary[opt].keys())
        test_cases = sorted(list(test_cases))
        
        # Print header
        print(f"\n{'Test Case':<20} {'Optimizer':<15} {'Mean Best':<15} {'Success':<10} "
              f"{'ERT':<12} {'Conv.Speed':<12} {'Robustness':<12}")
        print("-" * 106)
        
        for tc in test_cases:
            for opt in optimizers:
                if tc in self.metrics_summary[opt]:
                    m = self.metrics_summary[opt][tc]
                    print(f"{tc:<20} {opt:<15} {m['mean_best']:<15.6e} "
                          f"{m['success_rate']:<10.1f} "
                          f"{m['ert']:<12.1f} "
                          f"{m['convergence_speed']:<12.2f} "
                          f"{m['robustness']:<12.2f}")
            print("-" * 106)
    
    def run(self, verbose: bool = True) -> Tuple[Dict, Dict]:
        """Run the complete comparison pipeline."""
        if verbose:
            print("\n" + "="*80)
            print("STARTING CMA-ES VARIANTS COMPARISON")
            print("="*80)
        
        # Run comparison
        self.run_comparison(verbose=verbose)
        
        # Compute metrics
        if verbose:
            print("\n" + "="*80)
            print("COMPUTING METRICS...")
            print("="*80)
        self.compute_metrics()
        if verbose:
            print("✓ Metrics computed successfully")
        
        # Create plots
        if verbose:
            print("\n" + "="*80)
            print("CREATING COCO-STYLE CDF PLOTS...")
            print("="*80)
        self.create_plots()
        
        # Save results
        if verbose:
            print("\n" + "="*80)
            print("SAVING RESULTS...")
            print("="*80)
        self.save_results()
        
        # Print summary
        if verbose:
            print("\n" + "="*80)
            print("COMPARISON SUMMARY")
            print("="*80)
        self.print_summary()
        
        return self.results, self.metrics_summary


def parse_arguments():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description='CMA-ES Variants Comparison on BBOB Benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on functions 1-3 with 2D and 5D (default, no verbose output)
  python3 test_cmaes_variants.py --functions 1,2,3 --dimensions 2,5
  
  # Test all 24 BBOB functions in 10D with 30 runs (with verbose output)
  python3 test_cmaes_variants.py --functions 1-24 --dimensions 10 --n_runs 30 --verbose
  
  # Quick test on specific functions with detailed progress
  python3 test_cmaes_variants.py --functions 1,5,10,15,20,24 --dimensions 5 --n_runs 5 --max_evals 500 --verbose
  
  # Comprehensive test across multiple dimensions with verbose progress tracking
  python3 test_cmaes_variants.py --functions 1-24 --dimensions 2,5,10,20 --n_runs 50 --max_evals 2000 --verbose
        """
    )
    
    p.add_argument('--functions', type=str, default='1,2,3',
                   help='Function IDs (comma-separated or range, e.g., "1,2,3" or "1-24")')
    p.add_argument('--dimensions', type=str, default='2,5',
                   help='Dimensions (comma-separated, e.g., "2,5,10")')
    p.add_argument('--n_runs', type=int, default=10,
                   help='Number of runs per test case (default: 10)')
    p.add_argument('--max_evals', type=int, default=1000,
                   help='Maximum evaluations per run (default: 1000)')
    p.add_argument('--output_dir', type=str, default='results',
                   help='Output directory for results (default: results)')
    p.add_argument('--verbose', action='store_true',
                   help='Enable verbose output')
    
    return p.parse_args()


def parse_list_arg(s: str) -> List[int]:
    """Parse comma-separated list or range (e.g., '1,2,3' or '1-24')."""
    result = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(list(set(result)))


def main():
    """Main entry point."""
    args = parse_arguments()
    
    test_functions = parse_list_arg(args.functions)
    dimensions = parse_list_arg(args.dimensions)
    
    # Show configuration
    print("="*80)
    print("CMA-ES VARIANTS COMPREHENSIVE COMPARISON")
    print("="*80)
    print(f"Algorithms: CMA-ES, AFN-CMA-ES, LQ-CMA-ES, DTS-CMA-ES, LMM-CMA-ES")
    print(f"Test Functions: {test_functions}")
    print(f"Dimensions: {dimensions}")
    print(f"Runs per case: {args.n_runs}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"Output directory: {args.output_dir}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print("="*80)
    
    comparison = CMAESVariantsComparison(
        test_functions=test_functions,
        dimensions=dimensions,
        n_runs=args.n_runs,
        max_evaluations=args.max_evals,
        save_dir=args.output_dir,
    )
    
    comparison.run(verbose=args.verbose)
    
    print("\n" + "="*80)
    print("✓ CMA-ES variants comparison completed successfully!")
    print("="*80)
    print(f"\nResults saved to: {comparison.output_dir}")
    print("\nGenerated files:")
    print("  - runs.json                    (detailed results)")
    print("  - metrics_summary.json         (performance metrics)")
    print("  - config.json                  (experiment configuration)")
    print("  - cdf_1e-8.png                 (COCO CDF plot, target 1e-8)")
    print("  - cdf_1e-5.png                 (COCO CDF plot, target 1e-5)")
    print("  - cdf_1e-2.png                 (COCO CDF plot, target 1e-2)")
    print("  - cdf_multiple_targets.png     (COCO CDF plots, 4 targets)")
    print("  - performance_profile.png      (Performance profile plot)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

