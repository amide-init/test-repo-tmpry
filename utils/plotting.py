"""
Plotting functions for optimization comparison results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .helpers import extract_history


class ComparisonPlotter:
    """Create comparison plots for optimization algorithms"""
    
    def __init__(self, output_dir: str, algorithms: List[str] = None):
        """
        Initialize plotter
        
        Args:
            output_dir: Directory to save plots
            algorithms: List of algorithm names to plot (optional, auto-detected from results if None)
        """
        self.output_dir = output_dir
        
        # Algorithm display order and colors
        if algorithms is None:
            self.algorithms = ["AFN", "CMA-ES", "AFN-CMA-ES", "LQ-CMA-ES", "DTS-CMA-ES", "LMM-CMA-ES"]
        else:
            self.algorithms = algorithms
            
        # Color palette
        self.color_map = {
            'AFN': '#e74c3c',
            'CMA-ES': '#1f77b4', 
            'AFN-CMA-ES': '#ff7f0e',
            'LQ-CMA-ES': '#2ca02c',
            'DTS-CMA-ES': '#d62728',
            'LMM-CMA-ES': '#9467bd'
        }
        self.colors = [self.color_map.get(alg, f'#{"".join([f"{ord(c):02x}" for c in alg[:3]])}') 
                      for alg in self.algorithms]
        
        # COCO-style target precisions (powers of 10)
        self.target_precisions = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8]
        
    def plot_metric(self, metrics_summary: Dict[str, Dict], 
                   metric_key: str, 
                   title: str, 
                   ylabel: str, 
                   filename: str) -> None:
        """
        Create a bar chart comparing algorithms on a specific metric
        
        Args:
            metrics_summary: Dictionary of metrics for each algorithm
            metric_key: Key of metric to plot
            title: Plot title
            ylabel: Y-axis label
            filename: Output filename
        """
        # Collect all test cases
        test_cases = set()
        for alg in metrics_summary:
            test_cases.update(metrics_summary[alg].keys())
        test_cases = sorted(list(test_cases))
        
        if not test_cases:
            return
        
        # Create figure
        plt.figure(figsize=(max(14, len(test_cases) * 2), 8))
        x = np.arange(len(test_cases))
        width = 0.13
        
        # Plot bars for each algorithm
        for i, (alg, color) in enumerate(zip(self.algorithms, self.colors)):
            vals = [metrics_summary.get(alg, {}).get(tc, {}).get(metric_key, 0.0) 
                   for tc in test_cases]
            plt.bar(x + i * width, vals, width, label=alg, color=color, alpha=0.85)
        
        # Format plot
        plt.xlabel('Test Cases', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(x + width * 2.5, test_cases, rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        plt.tight_layout()
        
        # Save
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def compute_cdf_data(self, results: Dict[str, List[Dict]], 
                        target_precision: float = 1e-8,
                        optimum: float = 0.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute CDF data for COCO-style plots
        
        IMPORTANT: Budgets are expressed as multiples of dimension (e.g., 10×D, 50×D)
        following COCO benchmarking standards.
        
        For BBOB functions with unknown optimum, uses relative improvement from initial value.
        
        Args:
            results: Dictionary of results for each algorithm
            target_precision: Target precision (relative improvement factor, e.g., 1e-2 = 99% improvement)
            optimum: Known optimum value (if None, uses relative improvement)
            
        Returns:
            Dictionary mapping algorithm name to (evaluations_normalized, proportion_solved) arrays
        """
        cdf_data = {}
        
        for alg in self.algorithms:
            all_evals_to_target = []
            total_runs = 0
            
            # Collect data across all test cases
            for key, runs in results.items():
                if not key.endswith(f"_{alg}"):
                    continue
                
                # Extract dimension from key (e.g., "func1_dim10_AFN" -> 10)
                parts = key.split('_')
                dimension = None
                for part in parts:
                    if part.startswith('dim'):
                        dimension = int(part.replace('dim', ''))
                        break
                
                if dimension is None:
                    continue
                
                for run in runs:
                    total_runs += 1
                    history = extract_history(run)
                    if not history or len(history) < 2:
                        continue
                    
                    # Use relative improvement from initial value
                    initial_val = history[0]
                    best_val = min(history)
                    total_improvement = initial_val - best_val
                    
                    # Target: reach (1 - target_precision) of total improvement
                    # E.g., target_precision=0.01 means reach 99% of improvement
                    target_val = initial_val - (1.0 - target_precision) * total_improvement
                    
                    # Find first evaluation where target is reached
                    for eval_idx, val in enumerate(history):
                        if val <= target_val:
                            # Normalize by dimension: budget = evaluations / dimension
                            normalized_budget = (eval_idx + 1) / dimension
                            all_evals_to_target.append(normalized_budget)
                            break
            
            if all_evals_to_target:
                # Sort normalized budgets
                sorted_budgets = np.sort(all_evals_to_target)
                # Compute proportion of runs solved at each budget
                proportions = np.arange(1, len(sorted_budgets) + 1) / len(sorted_budgets)
                cdf_data[alg] = (sorted_budgets, proportions)
            else:
                # No runs reached target - return empty for proper handling in plot
                # Add a warning message
                print(f"WARNING: {alg} - 0/{total_runs} runs reached target precision {target_precision:.1e}")
                cdf_data[alg] = (np.array([]), np.array([]))
        
        return cdf_data
    
    def plot_cdf(self, results: Dict[str, List[Dict]], 
                 target_precision: float = 1e-8,
                 optimum: float = 0.0,
                 filename: str = 'cdf_plot.png',
                 title: str = None) -> None:
        """
        Create COCO-style CDF plot
        
        Budgets are expressed as multiples of dimension (e.g., 10×D, 50×D)
        following COCO benchmarking standards.
        
        Args:
            results: Dictionary of results for each algorithm
            target_precision: Target precision to reach
            optimum: Known optimum value
            filename: Output filename
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        cdf_data = self.compute_cdf_data(results, target_precision, optimum)
        
        # Check if any algorithm has data
        has_data = False
        for alg, color in zip(self.algorithms, self.colors):
            if alg not in cdf_data or len(cdf_data[alg][0]) == 0:
                continue
            
            has_data = True
            budgets, proportions = cdf_data[alg]
            plt.plot(budgets, proportions, color=color, label=alg, linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        
        # Budgets must be expressed as multiples of dimension
        plt.xlabel('Budget (# evaluations / dimension)', fontsize=14, fontweight='bold')
        plt.ylabel('Proportion of Problems Solved', fontsize=14, fontweight='bold')
        
        if title is None:
            improvement_pct = (1.0 - target_precision) * 100
            title = f'CDF: Reaching {improvement_pct:.1f}% of Total Improvement'
        plt.title(title, fontsize=16, fontweight='bold')
        
        if not has_data:
            # Add warning text to plot
            plt.text(0.5, 0.5, 
                    f'NO DATA: No runs reached target precision {target_precision:.1e}\n'
                    'Try: More evaluations or relaxed target precision',
                    transform=plt.gca().transAxes,
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            print(f"  WARNING: No data to plot for target precision {target_precision:.1e}")
        
        plt.xscale('log')
        plt.grid(True, which='both', alpha=0.3, linestyle='--')
        if has_data:
            plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
        plt.xlim(left=1)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def plot_multiple_target_cdf(self, results: Dict[str, List[Dict]],
                                 optimum: float = 0.0,
                                 filename: str = 'cdf_multiple_targets.png') -> None:
        """
        Create COCO-style CDF plot with multiple target precisions
        
        Budgets are expressed as multiples of dimension (e.g., 10×D, 50×D)
        following COCO benchmarking standards.
        
        Args:
            results: Dictionary of results for each algorithm
            optimum: Known optimum value
            filename: Output filename
        """
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
            improvement_pct = (1.0 - target) * 100
            ax.set_title(f'Target: {improvement_pct:.1f}% Improvement', fontsize=13, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, which='both', alpha=0.3, linestyle='--')
            ax.set_xlim(left=1)
            ax.set_ylim([0, 1.05])
            
            if idx == 3:  # Only show legend on last subplot
                ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
        
        plt.suptitle('COCO-style CDF at Multiple Target Precisions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def plot_performance_profile(self, results: Dict[str, List[Dict]],
                                 target_precision: float = 1e-8,
                                 optimum: float = 0.0,
                                 filename: str = 'performance_profile.png') -> None:
        """
        Create performance profile plot (ratio to best algorithm)
        
        Budgets are expressed as multiples of dimension (e.g., 10×D, 50×D)
        following COCO benchmarking standards.
        
        Args:
            results: Dictionary of results for each algorithm
            target_precision: Target precision to reach
            optimum: Known optimum value
            filename: Output filename
        """
        plt.figure(figsize=(12, 8))
        
        # Collect evaluations to target for each algorithm on each problem
        algo_evals = {alg: [] for alg in self.algorithms}
        
        # Group by problem (function + dimension)
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
            
            # Get median normalized budget to target for this algorithm on this problem
            budgets_to_target = []
            for run in runs:
                history = extract_history(run)
                if not history or len(history) < 2:
                    continue
                
                # Use relative improvement
                initial_val = history[0]
                best_val = min(history)
                total_improvement = initial_val - best_val
                target_val = initial_val - (1.0 - target_precision) * total_improvement
                
                for eval_idx, val in enumerate(history):
                    if val <= target_val:
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
            
            # Find best (minimum) budget across all algorithms for this problem
            min_budget = min(alg_results.values())
            for alg in self.algorithms:
                if alg in alg_results:
                    # Ratio of this algorithm's budget to the best budget
                    ratio = alg_results[alg] / min_budget
                    algo_evals[alg].append(ratio)
        
        # Plot performance profiles
        for alg, color in zip(self.algorithms, self.colors):
            if not algo_evals[alg]:
                continue
            
            ratios = np.sort(algo_evals[alg])
            proportions = np.arange(1, len(ratios) + 1) / len(ratios)
            plt.plot(ratios, proportions, color=color, label=alg, linewidth=2.5, alpha=0.8)
        
        plt.xlabel('Performance Ratio (relative to best)', fontsize=14, fontweight='bold')
        plt.ylabel('Proportion of Problems', fontsize=14, fontweight='bold')
        improvement_pct = (1.0 - target_precision) * 100
        plt.title(f'Performance Profile (Target: {improvement_pct:.1f}% Improvement)', fontsize=16, fontweight='bold')
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
    
    def create_all_plots(self, results: Dict[str, List[Dict]] = None, 
                        metrics_summary: Dict[str, Dict] = None,
                        optimum: float = 0.0) -> None:
        """
        Create all comparison plots (COCO-style CDF plots + convergence curves)
        
        Args:
            results: Dictionary of raw results for each algorithm (required for CDF plots)
            metrics_summary: Dictionary of metrics for each algorithm (optional, for legacy support)
            optimum: Known optimum value for problems
        """
        print("\nCreating Comparison Plots...")
        
        if results is not None:
            # Create convergence curves for each problem (ALWAYS shows data!)
            print("Creating convergence curves...")
            self.plot_all_convergence_curves(results, filename='convergence_curves.png')
            
            # Create COCO-style CDF plots
            self.plot_cdf(results, target_precision=1e-8, optimum=optimum, 
                         filename='cdf_1e-8.png')
            
            self.plot_cdf(results, target_precision=1e-5, optimum=optimum,
                         filename='cdf_1e-5.png')
            
            self.plot_cdf(results, target_precision=1e-2, optimum=optimum,
                         filename='cdf_1e-2.png')
            
            self.plot_multiple_target_cdf(results, optimum=optimum,
                                         filename='cdf_multiple_targets.png')
            
            self.plot_performance_profile(results, target_precision=1e-8, optimum=optimum,
                                         filename='performance_profile.png')
        
        print("All plots created successfully!")
    
    def plot_all_convergence_curves(self, results: Dict[str, List[Dict]],
                                    filename: str = 'convergence_curves.png') -> None:
        """
        Plot average convergence curves for all algorithms on all problems
        
        Args:
            results: Dictionary of results for each algorithm
            filename: Output filename
        """
        # Group results by problem
        problems = {}
        for key in results.keys():
            parts = key.rsplit('_', 1)
            if len(parts) < 2:
                continue
            problem_key = parts[0]
            if problem_key not in problems:
                problems[problem_key] = []
            problems[problem_key].append(key)
        
        n_problems = len(problems)
        if n_problems == 0:
            return
        
        # Create subplots
        n_cols = min(3, n_problems)
        n_rows = (n_problems + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_problems == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (problem_key, keys) in enumerate(sorted(problems.items())):
            ax = axes[idx]
            
            for key in keys:
                alg = key.rsplit('_', 1)[1]
                if alg not in self.algorithms:
                    continue
                    
                color = self.color_map.get(alg, '#000000')
                runs = results[key]
                
                # Get all histories
                histories = []
                for r in runs:
                    h = extract_history(r)
                    if h:
                        histories.append(h)
                
                if not histories:
                    continue
                
                # Compute mean and std
                max_len = max(len(h) for h in histories)
                padded = []
                for h in histories:
                    if len(h) < max_len:
                        h = list(h) + [h[-1]] * (max_len - len(h))
                    padded.append(h)
                
                mean_curve = np.mean(padded, axis=0)
                std_curve = np.std(padded, axis=0)
                x = np.arange(1, len(mean_curve) + 1)
                
                ax.plot(x, mean_curve, color=color, label=alg, linewidth=2, alpha=0.8)
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                               color=color, alpha=0.2)
            
            ax.set_xlabel('Function Evaluations', fontsize=10)
            ax.set_ylabel('Best Function Value', fontsize=10)
            ax.set_title(f'{problem_key.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_problems, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()
    
    def plot_convergence_curves(self, results: Dict[str, List[Dict]], 
                                func_id: int, 
                                dimension: int,
                                filename: str = None) -> None:
        """
        Plot convergence curves for all algorithms on a specific test case
        
        Args:
            results: Dictionary of results for each algorithm
            func_id: Function ID
            dimension: Problem dimension
            filename: Output filename (optional)
        """
        plt.figure(figsize=(12, 8))
        
        for i, (alg, color) in enumerate(zip(self.algorithms, self.colors)):
            key = f"func{func_id}_dim{dimension}_{alg}"
            if key not in results:
                continue
            
            runs = results[key]
            # Average convergence curve across runs
            histories = []
            for r in runs:
                h = r.get('history') or r.get('y_history') or []
                if isinstance(h, np.ndarray):
                    h = h.tolist()
                if h:
                    histories.append(h)
            
            if histories:
                # Pad histories to same length
                max_len = max(len(h) for h in histories)
                padded = []
                for h in histories:
                    if len(h) < max_len:
                        h = h + [h[-1]] * (max_len - len(h))
                    padded.append(h)
                
                mean_curve = np.mean(padded, axis=0)
                std_curve = np.std(padded, axis=0)
                x = np.arange(len(mean_curve))
                
                plt.plot(x, mean_curve, color=color, label=alg, linewidth=2)
                plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                               color=color, alpha=0.2)
        
        plt.xlabel('Function Evaluations', fontsize=12)
        plt.ylabel('Best Function Value', fontsize=12)
        plt.title(f'Convergence Curves - Function {func_id}, Dimension {dimension}', 
                 fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename is None:
            filename = f'convergence_f{func_id}_d{dimension}.png'
        
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

