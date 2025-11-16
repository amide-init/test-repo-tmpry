#!/usr/bin/env python3
"""
AFN vs GA vs PSO vs ACO Comparison
This runner uses the ensemble-based AFN implementation for comparison
with GA, PSO, and ACO algorithms. All outputs and plots are consistent
with the original runner so existing workflows don't break.
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
from typing import List, Tuple, Dict, Callable

# Add paths
sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

# Local imports
from afn.afn_core import AFNCore
from afn.comparison_algorithms import GA, PSO, ACO
from data.sample import load_bbob_function


class AFNGaPsoAcoComparison:
	"""Comprehensive comparison using ensemble-based AFN core."""
	def __init__(self,
				 test_functions: List[int] = [1, 2, 3],
				 dimensions: List[int] = [2, 5],
				 n_runs: int = 10,
				 max_evaluations: int = 100,
				 save_dir: str = "results",
				 model_type: str = "random_forest"):
		self.test_functions = test_functions
		self.dimensions = dimensions
		self.n_runs = n_runs
		self.max_evaluations = max_evaluations
		self.save_dir = save_dir
		self.model_type = model_type

		self.results: Dict[str, List[Dict]] = {}
		self.metrics_summary: Dict[str, Dict] = {}

		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		self.output_dir = os.path.join(save_dir, f'afn_ga_pso_aco_{timestamp}')
		os.makedirs(self.output_dir, exist_ok=True)
		print(f"Results will be saved to: {self.output_dir}")

	def run_single_test(self, func_id: int, dimension: int, algorithm_name: str, run_idx: int, seed: int = None) -> Dict:
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# Get COCO/BBOB function and bounds
		problem, info = load_bbob_function(func_id, dimension, instance=1)
		objective_function = problem
		bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(dimension)]

		# Instantiate algorithm
		if algorithm_name == 'AFN':
			algorithm = AFNCore(
				input_dim=dimension,
				bounds=bounds,
				uncertainty_threshold=0.03,
				batch_size=8,
				max_evaluations=self.max_evaluations,
				convergence_threshold=1e-6,
				convergence_window=10,
				model_type=self.model_type,
			)
		elif algorithm_name == 'GA':
			algorithm = GA(population_size=100, crossover_rate=0.8, mutation_rate=0.1, max_generations=self.max_evaluations, bounds=bounds)
		elif algorithm_name == 'PSO':
			algorithm = PSO(n_particles=50, w=0.7, c1=1.5, c2=2.0, max_iterations=self.max_evaluations, bounds=bounds)
		elif algorithm_name == 'ACO':
			algorithm = ACO(n_ants=30, evaporation_rate=0.1, alpha=1.0, beta=2.0, max_iterations=self.max_evaluations, bounds=bounds)
		else:
			raise ValueError(f"Unknown algorithm: {algorithm_name}")

		start_time = time.time()
		result = algorithm.optimize(objective_function, verbose=False)
		end_time = time.time()

		result['execution_time'] = end_time - start_time
		result['algorithm'] = algorithm_name
		result['function_id'] = func_id
		result['dimension'] = dimension
		result['run'] = run_idx

		# Derive evaluation_count if missing
		if 'evaluation_count' not in result or not result['evaluation_count']:
			hist = result.get('history') or result.get('y_history') or []
			result['evaluation_count'] = int(len(hist)) if isinstance(hist, (list, np.ndarray)) else int(self.max_evaluations)

		return result

	def run_comparison(self, verbose: bool = True):
		algorithms = ['AFN', 'GA', 'PSO', 'ACO']
		for func_id in self.test_functions:
			for dim in self.dimensions:
				# Get function name from COCO/BBOB
				problem, _ = load_bbob_function(func_id, dim, instance=1)
				func_name = problem.name if hasattr(problem, 'name') else f"BBOB Function {func_id}"
				if verbose:
					print(f"\nTesting {func_name} (ID={func_id}, dimension {dim})")
				for alg in algorithms:
					key = f"func{func_id}_dim{dim}_{alg}"
					self.results[key] = []
					for run_idx in range(self.n_runs):
						seed = 42 + func_id * 1000 + dim * 100 + run_idx + hash(alg) % 1000
						res = self.run_single_test(func_id, dim, alg, run_idx, seed)
						self.results[key].append(res)
		return self.results

	def compute_metrics(self) -> Dict:
		metrics = {}
		for key, runs in self.results.items():
			func_id = int(key.split('_')[0].replace('func', ''))
			dimension = int(key.split('_')[1].replace('dim', ''))
			alg_name = key.split('_')[2]

			best_values = [r['best_y'] for r in runs]
			exec_times = [r['execution_time'] for r in runs]
			eval_counts = [r.get('evaluation_count', self.max_evaluations) for r in runs]

			# Histories for baseline (avoid boolean ops on numpy arrays)
			histories = []
			for r in runs:
				h = []
				if isinstance(r, dict):
					if 'history' in r and r['history'] is not None:
						h = r['history']
					elif 'y_history' in r and r['y_history'] is not None:
						h = r['y_history']
				# Normalize to list
				if isinstance(h, np.ndarray):
					h = h.tolist()
				elif not isinstance(h, list):
					h = list(h) if h is not None else []
				histories.append(h)

			# Optimization Accuracy (iterations formula baseline)
			optimum = 0.0
			opt_acc = []
			for i, bv in enumerate(best_values):
				start_val = float(histories[i][0]) if i < len(histories) and histories[i] else float(bv)
				gap = max(1e-12, start_val - optimum)
				acc = 100.0 * (1.0 - (float(bv) - optimum) / gap)
				opt_acc.append(float(np.clip(acc, 0.0, 100.0)))

			# Convergence speed as percentage of evaluations to reach 95% of improvement
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

			test_key = f"func{func_id}_dim{dimension}"
			metrics.setdefault(alg_name, {})[test_key] = {
				'convergence_speed': float(np.mean(conv_speeds)) if conv_speeds else 0.0,
				'optimization_accuracy': float(np.mean(opt_acc)) if opt_acc else 0.0,
				'resource_utilization': float(np.mean([min(100.0, (c / max(1, self.max_evaluations)) * 100.0) for c in eval_counts])),
				'exploitation_balance_ratio': self._calculate_exploitation_balance(histories, alg_name),
				'robustness': float(max(0.0, (1.0 - (np.std(best_values) / max(1e-12, np.mean(best_values)))) * 100.0)) if len(best_values) > 1 else 100.0,
				'mean_best': float(np.mean(best_values)),
				'std_best': float(np.std(best_values)),
				'mean_time': float(np.mean(exec_times)),
				'n_runs': len(runs),
			}
		self.metrics_summary = metrics
		return metrics

	def _calculate_exploitation_balance(self, histories: List[List[float]], algorithm_name: str = "AFN") -> float:
		"""Calculate algorithm-specific exploitation balance using E/(E+R) formula (0-100%)"""
		if not histories:
			return 0.0
		
		balances = []
		for history in histories:
			if len(history) > 10:
				E, R = self._calculate_algorithm_contributions(history, algorithm_name)
				total = E + R
				if total > 0:
					balance = E / total
					# Ensure balance is in reasonable range
					balance = max(0.0, min(1.0, balance))
					balances.append(balance)
		
		return float(np.mean(balances) * 100) if balances else 50.0  # Default to 50% if no valid data
	
	def _calculate_algorithm_contributions(self, history: List[float], algorithm_name: str) -> tuple:
		"""Calculate algorithm-specific exploitation (E) and exploration (R) contributions"""
		
		if algorithm_name == "AFN":
			return self._calculate_afn_contributions(history)
		elif algorithm_name == "PSO":
			return self._calculate_pso_contributions(history)
		elif algorithm_name == "GA":
			return self._calculate_ga_contributions(history)
		elif algorithm_name == "ACO":
			return self._calculate_aco_contributions(history)
		else:
			# Fallback to generic calculation
			return self._calculate_generic_contributions(history)
	
	def _calculate_afn_contributions(self, history: List[float]) -> tuple:
		"""AFN-specific: E = surrogate model refinement, R = uncertainty-based sampling"""
		if len(history) < 5:
			return 0.0, 0.0
		
		E = 0.0  # Exploitation: local refinement efforts
		R = 0.0  # Exploration: uncertainty-driven sampling
		
		# Calculate variance and trend to determine exploitation vs exploration
		improvements = [history[i-1] - history[i] for i in range(1, len(history))]
		mean_improvement = np.mean(improvements) if improvements else 0.0
		improvement_variance = np.var(improvements) if len(improvements) > 1 else 0.0
		
		for i, improvement in enumerate(improvements):
			progress_factor = (i + 1) / len(improvements)
			
			if improvement > 0:  # Improving step
				# AFN exploitation: local refinement with adaptive weighting
				# More exploitation early, less as optimization progresses
				exploitation_weight = 0.6 + 0.4 * (1.0 - progress_factor)
				E += improvement * exploitation_weight
			else:
				# AFN exploration: uncertainty-driven sampling
				# More exploration needed when variance is high
				exploration_weight = 0.4 + 0.6 * min(1.0, improvement_variance / max(1.0, mean_improvement))
				R += abs(improvement) * exploration_weight
		
		# Ensure we don't get extreme values - simplified approach
		# Just return the raw E and R values, let the ratio calculation handle it
		
		return E, R
	
	def _calculate_pso_contributions(self, history: List[float]) -> tuple:
		"""PSO-specific: E = movement towards global best, R = diversity maintenance"""
		if len(history) < 5:
			return 0.0, 0.0
		
		E = 0.0  # Exploitation: global best attraction
		R = 0.0  # Exploration: swarm diversity
		
		improvements = [history[i-1] - history[i] for i in range(1, len(history))]
		mean_improvement = np.mean(improvements) if improvements else 0.0
		
		for i, improvement in enumerate(improvements):
			progress_factor = (i + 1) / len(improvements)
			
			if improvement > 0:  # Improving step
				# PSO exploitation: strong attraction to global best
				# High exploitation due to global best guidance, but decreases over time
				exploitation_weight = 0.8 + 0.2 * (1.0 - progress_factor)
				E += improvement * exploitation_weight
			else:
				# PSO exploration: inertia and random components
				# Moderate exploration, increases when convergence slows
				exploration_weight = 0.3 + 0.4 * progress_factor
				R += abs(improvement) * exploration_weight
		
		# Simplified approach - return raw values
		
		return E, R
	
	def _calculate_ga_contributions(self, history: List[float]) -> tuple:
		"""GA-specific: E = crossover improvement, R = mutation diversity"""
		if len(history) < 5:
			return 0.0, 0.0
		
		E = 0.0  # Exploitation: crossover-based improvement
		R = 0.0  # Exploration: mutation-driven diversity
		
		improvements = [history[i-1] - history[i] for i in range(1, len(history))]
		mean_improvement = np.mean(improvements) if improvements else 0.0
		
		for i, improvement in enumerate(improvements):
			progress_factor = (i + 1) / len(improvements)
			
			if improvement > 0:  # Improving step
				# GA exploitation: crossover creates offspring near parents
				# Balanced exploitation, increases with selection pressure
				exploitation_weight = 0.5 + 0.3 * progress_factor
				E += improvement * exploitation_weight
			else:
				# GA exploration: mutation introduces diversity
				# High exploration, especially early in evolution
				exploration_weight = 0.6 + 0.3 * (1.0 - progress_factor)
				R += abs(improvement) * exploration_weight
		
		# Simplified approach - return raw values
		
		return E, R
	
	def _calculate_aco_contributions(self, history: List[float]) -> tuple:
		"""ACO-specific: E = pheromone exploitation, R = exploration trails"""
		if len(history) < 5:
			return 0.0, 0.0
		
		E = 0.0  # Exploitation: pheromone trail following
		R = 0.0  # Exploration: random exploration
		
		improvements = [history[i-1] - history[i] for i in range(1, len(history))]
		mean_improvement = np.mean(improvements) if improvements else 0.0
		
		for i, improvement in enumerate(improvements):
			progress_factor = (i + 1) / len(improvements)
			
			if improvement > 0:  # Improving step
				# ACO exploitation: following strong pheromone trails
				# Moderate exploitation, increases as trails strengthen
				exploitation_weight = 0.4 + 0.4 * progress_factor
				E += improvement * exploitation_weight
			else:
				# ACO exploration: random exploration and trail discovery
				# High exploration, especially early in search
				exploration_weight = 0.7 + 0.2 * (1.0 - progress_factor)
				R += abs(improvement) * exploration_weight
		
		# Simplified approach - return raw values
		
		return E, R
	
	def _calculate_generic_contributions(self, history: List[float]) -> tuple:
		"""Generic fallback calculation"""
		if len(history) < 5:
			return 0.0, 0.0
		
		E = 0.0
		R = 0.0
		
		for i in range(1, len(history)):
			improvement = history[i-1] - history[i]
			
			if improvement > 0:
				E += improvement * 0.6  # Generic exploitation weight
			else:
				R += abs(improvement) * 0.6  # Generic exploration weight
		
		return E, R

	def _plot_metric(self, metric_key: str, title: str, ylabel: str, filename: str) -> None:
		test_cases = set()
		for alg in self.metrics_summary:
			test_cases.update(self.metrics_summary[alg].keys())
		test_cases = sorted(list(test_cases))
		if not test_cases:
			return
		plt.figure(figsize=(max(12, len(test_cases) * 1.8), 8))
		x = np.arange(len(test_cases))
		width = 0.2
		for i, (alg, color) in enumerate(zip(['AFN', 'GA', 'PSO', 'ACO'], ['steelblue', 'orange', 'green', 'red'])):
			vals = [self.metrics_summary.get(alg, {}).get(tc, {}).get(metric_key, 0.0) for tc in test_cases]
			plt.bar(x + i * width, vals, width, label=alg, color=color, alpha=0.85)
		plt.xlabel('Test Cases')
		plt.ylabel(ylabel)
		plt.title(title)
		plt.xticks(x + width * 1.5, test_cases, rotation=45)
		plt.grid(True, axis='y', alpha=0.3)
		plt.legend()
		plt.tight_layout()
		path = os.path.join(self.output_dir, filename)
		plt.savefig(path, dpi=300, bbox_inches='tight')
		print(f"  Saved: {path}")
		plt.close()

	def create_plots(self) -> None:
		print("\nCreating Comparison Plots...")
		self._plot_metric('convergence_speed', 'Convergence Speed', 'Convergence Speed (%)', 'convergence_speed.png')
		self._plot_metric('optimization_accuracy', 'Optimization Accuracy', 'Optimization Accuracy (%)', 'optimization_accuracy.png')
		self._plot_metric('resource_utilization', 'Resource Utilization', 'Resource Utilization (%)', 'resource_utilization.png')
		self._plot_metric('exploitation_balance_ratio', 'Exploitation Balance Ratio', 'Exploitation Balance Ratio (%)', 'exploitation_balance.png')
		self._plot_metric('robustness', 'Robustness of Optimization', 'Robustness (%)', 'robustness.png')

	def save_results(self) -> None:
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
				'algorithms': ['AFN', 'GA', 'PSO', 'ACO'],
				'timestamp': datetime.now().isoformat(),
			}, f, indent=2)
		print(f"Results saved to: {self.output_dir}")

	def run(self, verbose: bool = True) -> Tuple[Dict, Dict]:
		self.run_comparison(verbose=verbose)
		self.compute_metrics()
		self.create_plots()
		self.save_results()
		return self.results, self.metrics_summary


def parse_arguments():
	p = argparse.ArgumentParser(description='AFN vs GA/PSO/ACO comparison')
	p.add_argument('--functions', type=str, default='1,2,3', help='Comma-separated function IDs')
	p.add_argument('--dimensions', type=str, default='2,5', help='Comma-separated dimensions')
	p.add_argument('--n_runs', type=int, default=10, help='Runs per case')
	p.add_argument('--max_evals', type=int, default=100, help='Max evaluations per algorithm')
	p.add_argument('--model_type', type=str, default='random_forest', 
				   choices=['random_forest', 'mlp'],
				   help='Surrogate model type for AFN (random_forest or mlp) (default: random_forest)')
	p.add_argument('--output_dir', type=str, default='results', help='Output directory')
	p.add_argument('--verbose', action='store_true', help='Verbose output')
	return p.parse_args()


def parse_list_arg(s: str) -> List[int]:
	return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
	args = parse_arguments()
	test_functions = parse_list_arg(args.functions)
	dimensions = parse_list_arg(args.dimensions)
	
	print(f"\nAFN Surrogate Model: {args.model_type}")
	print(f"Functions: {test_functions}")
	print(f"Dimensions: {dimensions}")
	print(f"Runs per case: {args.n_runs}")
	print(f"Max evaluations: {args.max_evals}\n")
	
	cmp = AFNGaPsoAcoComparison(
		test_functions=test_functions,
		dimensions=dimensions,
		n_runs=args.n_runs,
		max_evaluations=args.max_evals,
		save_dir=args.output_dir,
		model_type=args.model_type,
	)
	cmp.run(verbose=args.verbose)
	print("\nAFN comparison completed!")


if __name__ == '__main__':
	sys.exit(main())
