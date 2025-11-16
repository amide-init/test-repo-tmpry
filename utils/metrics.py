"""
Metrics computation for optimization algorithm comparison
"""

import numpy as np
from typing import List, Dict, Any
from .helpers import extract_history


class MetricsCalculator:
    """Calculate performance metrics for optimization algorithms"""
    
    def __init__(self, max_evaluations: int):
        """
        Initialize metrics calculator
        
        Args:
            max_evaluations: Maximum number of function evaluations allowed
        """
        self.max_evaluations = max_evaluations
    
    def compute_success_rate(self, best_values: List[float], 
                             target_precision: float = 1e-8,
                             optimum: float = 0.0) -> float:
        """
        Compute success rate metric
        
        Success = reaching target precision: |f(x) - f*| <= target_precision
        
        Args:
            best_values: List of best values found
            target_precision: Target precision threshold
            optimum: Known optimum value
            
        Returns:
            Success rate as percentage (0-100)
        """
        successes = sum(1 for bv in best_values if abs(bv - optimum) <= target_precision)
        return 100.0 * successes / len(best_values) if best_values else 0.0
    
    def compute_ert(self, histories: List[list], 
                    target_precision: float = 1e-8,
                    optimum: float = 0.0) -> float:
        """
        Compute Expected Running Time (ERT)
        
        ERT = average number of evaluations needed to reach target precision
        For unsuccessful runs, uses max_evaluations as penalty
        
        Args:
            histories: List of optimization histories
            target_precision: Target precision threshold
            optimum: Known optimum value
            
        Returns:
            Expected Running Time (average evaluations to target)
        """
        evals_to_target = []
        
        for history in histories:
            if not history:
                evals_to_target.append(self.max_evaluations)
                continue
            
            # Find first evaluation where target precision is reached
            reached = False
            for eval_idx, val in enumerate(history):
                if abs(val - optimum) <= target_precision:
                    evals_to_target.append(eval_idx + 1)
                    reached = True
                    break
            
            if not reached:
                evals_to_target.append(self.max_evaluations)
        
        return float(np.mean(evals_to_target)) if evals_to_target else float(self.max_evaluations)
    
    def compute_convergence_speed(self, histories: List[list]) -> List[float]:
        """
        Compute convergence speed metric
        
        Measures percentage of budget used to reach 95% of final improvement
        
        Args:
            histories: List of optimization histories
            
        Returns:
            List of convergence speed percentages
        """
        conv_speeds = []
        
        for h in histories:
            if h and len(h) > 1:
                initial_val = h[0]
                final_val = min(h)
                total_improvement = initial_val - final_val
                
                if total_improvement > 1e-12:
                    # Target: 95% of improvement achieved
                    target_val = initial_val - 0.95 * total_improvement
                    for i, v in enumerate(h):
                        if v <= target_val:
                            # Convert to percentage of max evaluations, cap at 100%
                            conv_percentage = min(100.0, ((i + 1) / max(1, self.max_evaluations)) * 100.0)
                            conv_speeds.append(conv_percentage)
                            break
                    else:
                        # If never reached target, use 100% (used all evaluations)
                        conv_speeds.append(100.0)
                else:
                    # No significant improvement, convergence at first evaluation
                    conv_speeds.append(100.0 / max(1, self.max_evaluations))
            else:
                # No history or single value, assume immediate convergence
                conv_speeds.append(100.0 / max(1, self.max_evaluations))
        
        return conv_speeds
    
    def compute_resource_utilization(self, eval_counts: List[int]) -> float:
        """
        Compute resource utilization metric
        
        Measures percentage of evaluation budget actually used
        
        Args:
            eval_counts: List of evaluation counts used
            
        Returns:
            Average resource utilization percentage
        """
        utilizations = [min(100.0, (c / max(1, self.max_evaluations)) * 100.0) 
                       for c in eval_counts]
        return float(np.mean(utilizations))
    
    def compute_robustness(self, best_values: List[float]) -> float:
        """
        Compute robustness metric
        
        Measures consistency across multiple runs
        Formula: 100 × (1 - std(values) / mean(values))
        
        Args:
            best_values: List of best values from multiple runs
            
        Returns:
            Robustness percentage (higher is better)
        """
        if len(best_values) <= 1:
            return 100.0
        
        mean_val = np.mean(best_values)
        std_val = np.std(best_values)
        
        robustness = 100.0 * (1.0 - (std_val / max(1e-12, mean_val)))
        return float(max(0.0, robustness))
    
    def compute_all_metrics(self, runs: List[Dict[str, Any]], 
                           optimum: float = 0.0,
                           target_precision: float = 1e-8) -> Dict[str, float]:
        """
        Compute all performance metrics for a set of runs
        
        Args:
            runs: List of result dictionaries from optimization runs
            optimum: Known optimum value
            target_precision: Target precision for success rate and ERT
            
        Returns:
            Dictionary of computed metrics
        """
        # Extract data
        best_values = [r['best_y'] for r in runs if r['best_y'] != float('inf')]
        exec_times = [r['execution_time'] for r in runs]
        eval_counts = [r.get('evaluation_count', self.max_evaluations) for r in runs]
        
        if not best_values:
            return {}
        
        # Extract histories
        histories = [extract_history(r) for r in runs]
        
        # Compute metrics
        success_rate = self.compute_success_rate(best_values, target_precision, optimum)
        ert = self.compute_ert(histories, target_precision, optimum)
        conv_speeds = self.compute_convergence_speed(histories)
        resource_util = self.compute_resource_utilization(eval_counts)
        robustness = self.compute_robustness(best_values)
        
        return {
            'success_rate': success_rate,
            'ert': ert,
            'convergence_speed': float(np.mean(conv_speeds)) if conv_speeds else 0.0,
            'resource_utilization': resource_util,
            'robustness': robustness,
            'mean_best': float(np.mean(best_values)),
            'std_best': float(np.std(best_values)),
            'median_best': float(np.median(best_values)),
            'mean_time': float(np.mean(exec_times)),
            'n_runs': len(best_values),
        }

