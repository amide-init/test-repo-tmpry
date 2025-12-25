#!/usr/bin/env python3
"""
AFN vs AFN-CMA-ES Comparison
Compares the AFN algorithm with its CMA-ES-based variant using surrogate modeling:
- AFN: Standalone AFN algorithm
- AFN-CMA-ES: CMA-ES with AFN ensemble surrogate
"""

import argparse
import sys
import os
import json
import time
import glob
from datetime import datetime
from typing import List, Dict

# Add paths
sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

# COCO imports
try:
    import cocoex
    from cocoex import Observer
    import cocopp
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Warning: COCO packages not fully available. Install with: pip install coco-experiment cocopp")

# Local imports (only for truly novel algorithms)
from afn.afn_core import AFNCore
from afn.cmaes_variants import (
    AFN_CMA
)
from data.sample import load_bbob_function
from utils import MetricsCalculator, convert_to_json, parse_list_arg, setup_seeds, generate_seed


class OptimizationComparison:
    """Orchestrates comparison between AFN and AFN-CMA-ES"""
    
    # Supported algorithms for this script
    ALL_ALGORITHMS = ["AFN", "AFN-CMA-ES"]
    
    # Algorithms available in COCO archive (use existing datasets, don't run new experiments)
    # For this script we only use novel algorithms, so this list is intentionally empty.
    COCO_ARCHIVE_ALGORITHMS = []
    
    # Novel algorithms (run new experiments with COCO observers)
    # Only truly novel algorithms should use local implementations
    NOVEL_ALGORITHMS = ["AFN", "AFN-CMA-ES"]
    
    def __init__(self,
                 test_functions: List[int] = [1, 2, 3],
                 dimensions: List[int] = [2, 5],
                 n_runs: int = 10,
                 max_evaluations: int = 200,
                 save_dir: str = "results",
                 algorithms: List[str] = None,
                 model_type: str = "random_forest"):
        """
        Initialize comparison framework
        
        Args:
            test_functions: List of BBOB function IDs to test
            dimensions: List of problem dimensions
            n_runs: Number of independent runs per test case
            max_evaluations: Maximum function evaluations per run
            save_dir: Directory to save results
            algorithms: List of algorithm names to run
            model_type: Surrogate model type ('random_forest' or 'mlp')
        """
        self.test_functions = test_functions
        self.dimensions = dimensions
        self.n_runs = n_runs
        self.max_evaluations = max_evaluations
        self.save_dir = save_dir
        self.model_type = model_type
        
        # Set algorithms to run
        if algorithms is None:
            self.algorithms = ["AFN", "CMA-ES"]  # Default
        else:
            self.algorithms = algorithms
        
        # Validate algorithms
        for alg in self.algorithms:
            if alg not in self.ALL_ALGORITHMS:
                raise ValueError(f"Unknown algorithm: {alg}. Available: {', '.join(self.ALL_ALGORITHMS)}")
        
        # Initialize utilities
        self.metrics_calculator = MetricsCalculator(max_evaluations)
        
        # Results storage
        self.results: Dict[str, List[Dict]] = {}
        self.metrics_summary: Dict[str, Dict] = {}
        
        # Setup output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(save_dir, f'cmaes_comparison_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create COCO-compatible output directory structure
        self.coco_output_dir = os.path.join(self.output_dir, 'coco_logs')
        os.makedirs(self.coco_output_dir, exist_ok=True)
        
        # Directory for COCO archive datasets (for baseline algorithms)
        self.coco_archive_dir = os.path.join(self.output_dir, 'coco_archive')
        os.makedirs(self.coco_archive_dir, exist_ok=True)
        
        print(f"Results will be saved to: {self.output_dir}")
        print(f"COCO logs (new experiments) will be saved to: {self.coco_output_dir}")
        print(f"COCO archive datasets (baselines) will be in: {self.coco_archive_dir}")
    
    def instantiate_algorithm(self, algorithm_name: str, dimension: int, 
                             bounds: List, seed: int):
        """
        Instantiate an optimization algorithm (only for novel algorithms)
        
        Note: Baseline algorithms (CMA-ES, LQ-CMA-ES) should use COCO archive datasets,
        not local implementations.
        
        Args:
            algorithm_name: Name of algorithm
            dimension: Problem dimension
            bounds: Search space bounds
            seed: Random seed
            
        Returns:
            Instantiated algorithm object
        """
        if algorithm_name in self.COCO_ARCHIVE_ALGORITHMS:
            raise ValueError(
                f"Algorithm '{algorithm_name}' should use COCO archive datasets, "
                f"not local implementation. Use load_coco_archive_data() instead."
            )
        
        if algorithm_name == "AFN":
            # Original standalone AFN
            return AFNCore(
                input_dim=dimension,
                bounds=bounds,
                uncertainty_threshold=0.03,
                batch_size=8,
                max_evaluations=self.max_evaluations,
                convergence_threshold=1e-6,
                convergence_window=10,
                model_type=self.model_type,
                random_state=seed
            )
        
        # Only truly novel algorithms use local implementations
        if algorithm_name == "AFN-CMA-ES":
            return AFN_CMA(
                bounds=bounds,
                max_evaluations=self.max_evaluations,
                model_type=self.model_type,
                random_state=seed
            )
        else:
            raise ValueError(
                f"Algorithm '{algorithm_name}' is not a novel algorithm. "
                f"Only AFN and AFN-CMA-ES should use local implementations. "
                f"Other algorithms should use COCO archive datasets."
            )
    
    def load_coco_archive_data(self, algorithm_name: str, func_id: int, 
                               dimension: int, verbose: bool = False) -> List[Dict]:
        """
        Load existing COCO archive datasets for baseline algorithms
        
        Args:
            algorithm_name: Name of algorithm (must be in COCO_ARCHIVE_ALGORITHMS)
            func_id: BBOB function ID
            dimension: Problem dimension
            verbose: Whether to print progress
            
        Returns:
            List of result dictionaries from COCO archive
        """
        if algorithm_name not in self.COCO_ARCHIVE_ALGORITHMS:
            raise ValueError(
                f"Algorithm '{algorithm_name}' is not in COCO archive. "
                f"Use run_single_test() for novel algorithms."
            )
        
        if not COCO_AVAILABLE:
            raise ImportError("COCO packages required to load archive data")
        
        if verbose:
            print(f"    Loading COCO archive data for {algorithm_name}...")
        
        # Map algorithm names to COCO archive names
        archive_name_map = {
            "CMA-ES": "CMA-ES",  # Standard name in COCO archive
            "LQ-CMA-ES": "CMA-ES-LQ",  # Linear-Quadratic variant (Hansen et al.)
            "DTS-CMA-ES": "CMA-ES-DTS",  # Dynamic Threshold Selection (Bajer et al.)
            "LMM-CMA-ES": "CMA-ES-LMM",  # Local Meta-Model (Loshchilov et al.)
            "BIPOP-CMA-ES": "BIPOP-CMA-ES"  # BIPOP variant (Hansen, 2009)
        }
        
        coco_alg_name = archive_name_map.get(algorithm_name, algorithm_name)
        
        # Find matching .dat files in COCO archive directory
        # Check multiple locations:
        # 1. results/.../coco_archive/ (per-run directory)
        # 2. data/coco-archive/ (shared archive directory)
        archive_locations = [
            self.coco_archive_dir,  # Per-run directory
            os.path.join('data', 'coco-archive'),  # Shared archive directory
        ]
        
        dat_files = []
        
        # Try standard format first: algorithm_bbob_f001_i01_d02.dat
        for archive_dir in archive_locations:
            pattern1 = os.path.join(
                archive_dir,
                f"{coco_alg_name}_bbob_f{func_id:03d}_i*_d{dimension:02d}.dat"
            )
            found_files = glob.glob(pattern1)
            if found_files:
                dat_files.extend(found_files)
                break
        
        # Try bbobexp format if standard format not found
        if not dat_files:
            for archive_dir in archive_locations:
                # Check for bbobexp format: AlgorithmName/data_fN/bbobexp_fN_DIMd.dat
                bbobexp_pattern = os.path.join(
                    archive_dir,
                    coco_alg_name,
                    f"data_f{func_id}",
                    f"bbobexp_f{func_id}_DIM{dimension}.dat"
                )
                if os.path.exists(bbobexp_pattern):
                    dat_files = [bbobexp_pattern]
                    break
        
        if not dat_files:
            if verbose:
                print(f"    ⚠ No .dat files found matching patterns:")
                print(f"      Checked locations:")
                for archive_dir in archive_locations:
                    print(f"        - {archive_dir}")
                    print(f"          Standard: {os.path.join(archive_dir, f'{coco_alg_name}_bbob_f{func_id:03d}_i*_d{dimension:02d}.dat')}")
                    print(f"          bbobexp: {os.path.join(archive_dir, coco_alg_name, f'data_f{func_id}', f'bbobexp_f{func_id}_DIM{dimension}.dat')}")
                print(f"    Please download datasets from COCO archive:")
                print(f"    https://numbbo.github.io/data-archive/")
                print(f"    Place .dat files in: {self.coco_archive_dir} or data/coco-archive/")
            return []
        
        if verbose:
            print(f"    Found {len(dat_files)} .dat file(s)")
        
        # Parse .dat files and convert to result format
        results = []
        import numpy as np
        
        for dat_file in sorted(dat_files):
            try:
                filename = os.path.basename(dat_file)
                is_bbobexp_format = 'bbobexp' in filename.lower()
                
                if is_bbobexp_format:
                    # Parse bbobexp format: multiple runs per file
                    # Format: % header line, then evaluation | fitness - Fopt | best fitness - Fopt | ...
                    # Each run starts with a % comment line
                    runs_data = []
                    current_run = {'evaluations': [], 'function_values': []}
                    
                    with open(dat_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            if line.startswith('%'):
                                # New run starts - save previous run if it has data
                                if current_run['evaluations']:
                                    runs_data.append(current_run)
                                    current_run = {'evaluations': [], 'function_values': []}
                                continue
                            
                            # Parse data line: evaluation fitness - Fopt best fitness - Fopt ...
                            # Format is space-separated (not pipe-separated like the header)
                            # Column 0: evaluation number
                            # Column 1: noise-free fitness - Fopt (this is the function value we want)
                            # Note: values may have + or - prefix (e.g., +8.463928927e+00)
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    eval_num = int(float(parts[0].strip()))
                                    # Column 1 is "noise-free fitness - Fopt", which is the function value
                                    # Handle + prefix (e.g., +8.463928927e+00)
                                    func_val_str = parts[1].strip()
                                    func_val = float(func_val_str)
                                    current_run['evaluations'].append(eval_num)
                                    current_run['function_values'].append(func_val)
                                except (ValueError, IndexError) as e:
                                    if verbose:
                                        print(f"      Warning: Failed to parse line: {line[:50]}... (Error: {e})")
                                    continue
                    
                    # Don't forget the last run
                    if current_run['evaluations']:
                        runs_data.append(current_run)
                    
                    if verbose and not runs_data:
                        print(f"      Warning: No runs extracted from {filename}. File may be empty or in unexpected format.")
                    
                    # Create results for each run
                    for run_idx, run_data in enumerate(runs_data):
                        if run_data['evaluations'] and run_data['function_values']:
                            result = {
                                'best_y': min(run_data['function_values']),
                                'history': run_data['function_values'],
                                'y_history': run_data['function_values'],
                                'evaluation_count': len(run_data['evaluations']),
                                'execution_time': 0.0,
                                'algorithm': algorithm_name,
                                'function_id': func_id,
                                'dimension': dimension,
                                'run': run_idx,  # 0-based run index
                                'best_x': None,
                                'converged': False
                            }
                            results.append(result)
                            
                            if verbose:
                                print(f"      Loaded run {run_idx + 1} from {filename} "
                                      f"({len(run_data['evaluations'])} evaluations, best={result['best_y']:.6e})")
                
                else:
                    # Parse standard COCO .dat file format
                    # Format: % evaluation_number function_value
                    # Lines starting with % are comments
                    evaluations = []
                    function_values = []
                    
                    with open(dat_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('%'):
                                continue
                            
                            # Parse data line: evaluation_number function_value
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    eval_num = int(float(parts[0]))
                                    func_val = float(parts[1])
                                    evaluations.append(eval_num)
                                    function_values.append(func_val)
                                except (ValueError, IndexError):
                                    continue
                    
                    if evaluations and function_values:
                        # Extract instance number from filename
                        # Format: algorithm_bbob_f001_i01_d02.dat
                        instance = 1
                        try:
                            parts = filename.split('_')
                            for part in parts:
                                if part.startswith('i'):
                                    instance = int(part[1:])
                                    break
                        except:
                            pass
                        
                        # Create result dictionary matching format from run_single_test
                        result = {
                            'best_y': min(function_values),
                            'history': function_values,
                            'y_history': function_values,
                            'evaluation_count': len(evaluations),
                            'execution_time': 0.0,  # Not available in archive data
                            'algorithm': algorithm_name,
                            'function_id': func_id,
                            'dimension': dimension,
                            'run': instance - 1,  # Convert instance to 0-based run index
                            'best_x': None,  # Not available in .dat files
                            'converged': False
                        }
                        results.append(result)
                        
                        if verbose:
                            print(f"      Loaded: {filename} "
                                  f"({len(evaluations)} evaluations, best={result['best_y']:.6e})")
            
            except Exception as e:
                if verbose:
                    print(f"      ✗ Error loading {os.path.basename(dat_file)}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        if verbose and results:
            print(f"    ✓ Successfully loaded {len(results)} runs from COCO archive")
        
        return results
    
    def _copy_archive_dat_files_to_coco_logs(self, algorithm_name: str, 
                                             func_id: int, dimension: int, 
                                             verbose: bool = False):
        """
        Copy COCO archive .dat files to coco_logs directory for cocopp processing
        
        This ensures all algorithms (both archive and novel) are in one directory
        so cocopp can process them together for fair comparison.
        """
        import shutil
        
        # Map algorithm names to COCO archive names
        archive_name_map = {
            "CMA-ES": "CMA-ES",
            "LQ-CMA-ES": "CMA-ES-LQ",
            "DTS-CMA-ES": "CMA-ES-DTS",
            "LMM-CMA-ES": "CMA-ES-LMM",
            "BIPOP-CMA-ES": "BIPOP-CMA-ES"
        }
        
        coco_alg_name = archive_name_map.get(algorithm_name, algorithm_name)
        
        # Find .dat files in archive directory
        # Check multiple locations: results/.../coco_archive/ and data/coco-archive/
        archive_locations = [
            self.coco_archive_dir,  # Per-run directory
            os.path.join('data', 'coco-archive'),  # Shared archive directory
        ]
        
        dat_files = []
        
        # Try standard format first
        for archive_dir in archive_locations:
            pattern = os.path.join(
                archive_dir,
                f"{coco_alg_name}_bbob_f{func_id:03d}_i*_d{dimension:02d}.dat"
            )
            found_files = glob.glob(pattern)
            if found_files:
                dat_files.extend(found_files)
                break
        
        # Try bbobexp format if standard format not found
        if not dat_files:
            for archive_dir in archive_locations:
                bbobexp_file = os.path.join(
                    archive_dir,
                    coco_alg_name,
                    f"data_f{func_id}",
                    f"bbobexp_f{func_id}_DIM{dimension}.dat"
                )
                if os.path.exists(bbobexp_file):
                    dat_files = [bbobexp_file]
                    break
        
        if not dat_files:
            return
        
        # Copy to coco_logs directory
        # For bbobexp format, preserve directory structure with algorithm name
        copied_count = 0
        is_bbobexp = any('bbobexp' in os.path.basename(f).lower() for f in dat_files)
        
        if is_bbobexp:
            # For bbobexp format: copy entire algorithm directory structure
            # Structure: AlgorithmName/data_fN/bbobexp_fN_DIMd.dat and AlgorithmName/bbobexp_fN.info
            source_alg_dir = os.path.dirname(os.path.dirname(dat_files[0]))  # Go up from data_fN to AlgorithmName
            dest_alg_dir = os.path.join(self.coco_output_dir, coco_alg_name)
            os.makedirs(dest_alg_dir, exist_ok=True)
            
            # Copy .info files for this function
            info_file = os.path.join(source_alg_dir, f"bbobexp_f{func_id}.info")
            if os.path.exists(info_file):
                dest_info = os.path.join(dest_alg_dir, f"bbobexp_f{func_id}.info")
                if not os.path.exists(dest_info):
                    shutil.copy2(info_file, dest_info)
                    copied_count += 1
                    if verbose:
                        print(f"      Copied: bbobexp_f{func_id}.info → {coco_alg_name}/")
            
            # Copy data directory structure
            source_data_dir = os.path.dirname(dat_files[0])  # data_fN directory
            dest_data_dir = os.path.join(dest_alg_dir, os.path.basename(source_data_dir))
            os.makedirs(dest_data_dir, exist_ok=True)
            
            # Copy all .dat and .tdat files for this function and dimension
            for dat_file in dat_files:
                try:
                    filename = os.path.basename(dat_file)
                    dest_path = os.path.join(dest_data_dir, filename)
                    if not os.path.exists(dest_path):
                        shutil.copy2(dat_file, dest_path)
                        copied_count += 1
                        if verbose:
                            print(f"      Copied: {filename} → {coco_alg_name}/{os.path.basename(dest_data_dir)}/")
                except Exception as e:
                    if verbose:
                        print(f"      ⚠ Failed to copy {os.path.basename(dat_file)}: {e}")
        else:
            # Standard format: copy .dat files directly
            for dat_file in dat_files:
                try:
                    filename = os.path.basename(dat_file)
                    dest_path = os.path.join(self.coco_output_dir, filename)
                    
                    # Only copy if not already exists (avoid overwriting)
                    if not os.path.exists(dest_path):
                        shutil.copy2(dat_file, dest_path)
                        copied_count += 1
                        if verbose:
                            print(f"      Copied: {filename} → coco_logs/")
                except Exception as e:
                    if verbose:
                        print(f"      ⚠ Failed to copy {os.path.basename(dat_file)}: {e}") 
        
        if verbose and copied_count > 0:
            print(f"    ✓ Copied {copied_count} file(s) to coco_logs/ for cocopp processing")
    
    def run_single_test(self, func_id: int, dimension: int, 
                       algorithm_name: str, run_idx: int, seed: int) -> Dict:
        """
        Run a single optimization test with COCO observer
        
        Args:
            func_id: BBOB function ID
            dimension: Problem dimension
            algorithm_name: Name of algorithm to test
            run_idx: Run index (for tracking)
            seed: Random seed
            
        Returns:
            Result dictionary
        """
        # Set random seeds
        setup_seeds(seed)
        
        # Load BBOB function
        problem, _ = load_bbob_function(func_id, dimension, instance=run_idx + 1)
        
        # Attach COCO observer BEFORE optimization (required for COCO compliance)
        observer = None
        if COCO_AVAILABLE:
            # Create observer with proper algorithm name for COCO
            # COCO expects algorithm names without special characters
            coco_alg_name = algorithm_name.replace('-', '_').replace(' ', '_')
            
            # Observer API: Observer(suite_name, result_folder_string)
            # The result_folder_string format: "result_folder: path/to/folder"
            # Algorithm name can be set via observer.set_algorithm_name() or in folder structure
            try:
                # Try standard Observer constructor (2 parameters)
                observer = Observer("bbob", f"result_folder: {self.coco_output_dir}")
                # Set algorithm name if method exists
                if hasattr(observer, 'set_algorithm_name'):
                    observer.set_algorithm_name(coco_alg_name)
                elif hasattr(observer, 'algorithm_name'):
                    observer.algorithm_name = coco_alg_name
            except TypeError:
                # If 2-param constructor fails, try 3-param (suite, folder, algorithm)
                try:
                    observer = Observer("bbob", f"result_folder: {self.coco_output_dir}", coco_alg_name)
                except TypeError:
                    # Fallback: try with just suite and folder, algorithm in folder name
                    alg_folder = os.path.join(self.coco_output_dir, coco_alg_name)
                    os.makedirs(alg_folder, exist_ok=True)
                    observer = Observer("bbob", f"result_folder: {alg_folder}")
            
            # Attach observer to problem
            problem.observe_with(observer)
        
        # Use the observed problem as objective function
        objective_function = problem
        bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) 
                 for i in range(dimension)]
        
        # Instantiate algorithm
        algorithm = self.instantiate_algorithm(algorithm_name, dimension, bounds, seed)
        
        # Run optimization
        start_time = time.time()
        result = algorithm.optimize(objective_function, verbose=False)
        end_time = time.time()
        
        # Finalize COCO observer (closes log files)
        # Note: Some cocoex versions don't have finalize() method
        # Files are automatically closed when observer goes out of scope
        if observer is not None:
            if hasattr(observer, 'finalize'):
                observer.finalize()
            # If finalize() doesn't exist, files will be closed automatically
            # when observer is garbage collected or problem is done
        
        # Add metadata
        result['execution_time'] = end_time - start_time
        result['algorithm'] = algorithm_name
        result['function_id'] = func_id
        result['dimension'] = dimension
        result['run'] = run_idx
        
        # Ensure evaluation_count is present
        if 'evaluation_count' not in result or not result['evaluation_count']:
            hist = result.get('history') or result.get('y_history') or []
            result['evaluation_count'] = len(hist) if hist else self.max_evaluations
        
        return result
    
    def run_comparison(self, verbose: bool = True) -> Dict:
        """
        Run full comparison across all test cases
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of all results
        """
        import numpy as np
        
        total_tests = len(self.test_functions) * len(self.dimensions)
        test_counter = 0
        
        for func_id in self.test_functions:
            for dim in self.dimensions:
                test_counter += 1
                
                # Get function info
                problem, _ = load_bbob_function(func_id, dim, instance=1)
                func_name = problem.name if hasattr(problem, 'name') else f"BBOB Function {func_id}"
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"[{test_counter}/{total_tests}] Testing {func_name} (ID={func_id}, dimension={dim})")
                    print('='*80)
                
                for alg_idx, alg in enumerate(self.algorithms, 1):
                    key = f"func{func_id}_dim{dim}_{alg}"
                    self.results[key] = []
                    
                    if verbose:
                        print(f"\n  [{alg_idx}/{len(self.algorithms)}] {alg}:")
                    
                    # Check if algorithm is in COCO archive (use existing datasets)
                    if alg in self.COCO_ARCHIVE_ALGORITHMS:
                        if verbose:
                            print(f"    Using COCO archive datasets for baseline algorithm: {alg}")
                        
                        try:
                            archive_results = self.load_coco_archive_data(alg, func_id, dim, verbose)
                            if archive_results:
                                self.results[key].extend(archive_results)
                                
                                # Copy .dat files to coco_logs so cocopp can process them
                                # This ensures all algorithms (archive + novel) are in one place
                                self._copy_archive_dat_files_to_coco_logs(alg, func_id, dim, verbose)
                                
                                if verbose:
                                    print(f"    ✓ Loaded {len(archive_results)} runs from COCO archive")
                            else:
                                if verbose:
                                    print(f"    ⚠ No archive data found. Please download from COCO archive.")
                                    print(f"    See: https://github.com/numbbo/coco/tree/master/data-archive")
                                    print(f"    Place .dat files in: {self.coco_archive_dir}")
                        except Exception as e:
                            print(f"    ✗ Failed to load COCO archive data: {e}")
                            import traceback
                            if verbose:
                                traceback.print_exc()
                    else:
                        # Novel algorithm - run new experiments with COCO observers
                        if verbose:
                            print(f"    Running new experiments with COCO observers...")
                        
                        for run_idx in range(self.n_runs):
                            seed = generate_seed(func_id, dim, run_idx, alg)
                            
                            try:
                                result = self.run_single_test(func_id, dim, alg, run_idx, seed)
                                self.results[key].append(result)
                                
                                if verbose:
                                    print(f"    Run {run_idx+1:2d}/{self.n_runs}: "
                                          f"best={result['best_y']:12.6e}, "
                                          f"evals={result.get('evaluation_count', 0):4d}, "
                                          f"time={result.get('execution_time', 0):6.2f}s")
                            
                            except Exception as e:
                                print(f"    ✗ Run {run_idx+1}/{self.n_runs} failed: {e}")
                                # Add failed result
                                self.results[key].append({
                                    'best_y': float('inf'),
                                    'history': [],
                                    'evaluation_count': 0,
                                    'execution_time': 0,
                                    'algorithm': alg,
                                    'function_id': func_id,
                                    'dimension': dim,
                                    'run': run_idx
                                })
                    
                    # Show summary for this algorithm
                    if verbose:
                        valid_results = [r for r in self.results[key] if r.get('best_y', float('inf')) != float('inf')]
                        if valid_results:
                            best_vals = [r['best_y'] for r in valid_results]
                            mean_best = np.mean(best_vals)
                            std_best = np.std(best_vals)
                            min_best = np.min(best_vals)
                            print(f"    → Summary: mean={mean_best:.6e}, std={std_best:.6e}, min={min_best:.6e}")
        
        return self.results
    
    def compute_metrics(self) -> Dict:
        """
        Compute performance metrics for all results
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        for key, runs in self.results.items():
            # Parse key
            parts = key.split('_')
            func_id = int(parts[0].replace('func', ''))
            dimension = int(parts[1].replace('dim', ''))
            alg_name = '_'.join(parts[2:])
            
            # Compute metrics using calculator
            test_metrics = self.metrics_calculator.compute_all_metrics(runs, optimum=0.0)
            
            if test_metrics:
                test_key = f"func{func_id}_dim{dimension}"
                metrics.setdefault(alg_name, {})[test_key] = test_metrics
        
        self.metrics_summary = metrics
        return metrics
    
    def save_results(self, save_json: bool = False) -> None:
        """
        Save results (optional JSON files for convenience, not required for COCO)
        
        Note: COCO compliance only requires .dat files (generated by observers).
        JSON files are optional convenience files for quick inspection.
        
        Args:
            save_json: Whether to save optional JSON files (default: False for COCO compliance)
        """
        if save_json:
            # Optional: Save detailed results in JSON format (not COCO standard)
            with open(os.path.join(self.output_dir, 'runs.json'), 'w') as f:
                json.dump(convert_to_json(self.results), f, indent=2)
            
            # Optional: Save metrics summary in JSON format (not COCO standard)
            with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
                json.dump(convert_to_json(self.metrics_summary), f, indent=2)
            
            # Optional: Save configuration
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump({
                    'test_functions': self.test_functions,
                    'dimensions': self.dimensions,
                    'n_runs': self.n_runs,
                    'max_evaluations': self.max_evaluations,
                    'algorithms': self.algorithms,
                    'model_type': self.model_type,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)
            
            print(f"\nOptional JSON files saved to: {self.output_dir}")
        
        # COCO standard format: .dat files are generated by observers in coco_logs/
        print(f"\nCOCO-compliant results (.dat files) in: {self.coco_output_dir}")
    
    def run(self, verbose: bool = True):
        """
        Run complete comparison pipeline
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Tuple of (results, metrics)
        """
        if verbose:
            print("\n" + "="*80)
            print("STARTING AFN vs CMA-ES VARIANTS COMPARISON")
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
        
        # Save results (COCO .dat files are already saved by observers)
        # JSON files are optional and not required for COCO compliance
        if verbose:
            print("\n" + "="*80)
            print("COCO RESULTS ALREADY SAVED...")
            print("="*80)
            print("Note: COCO .dat files are automatically generated by observers")
            print("      JSON files are optional (not COCO standard)")
        self.save_results(save_json=False)  # Set to True if you want optional JSON files
        
        return self.results, self.metrics_summary


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare AFN with AFN-CMA-ES on BBOB benchmark functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run AFN with MLP surrogate (default)
  python3 run_afn.py --algorithms AFN --functions 1,2,3 --dimensions 2,5
  
  # Run AFN with Random Forest surrogate
  python3 run_afn.py --algorithms AFN --model_type random_forest --functions 1,2,3 --dimensions 2,5
  
  # Run AFN and AFN-CMA-ES
  python3 run_afn.py --algorithms AFN,AFN-CMA-ES --functions 1,2,3 --dimensions 2,5
  
  # Run only AFN-CMA-ES (AFN with CMA-ES)
  python3 run_afn.py --algorithms AFN-CMA-ES --functions 1,2,3 --dimensions 2,5 --max_evals 500
  
  # Run all available algorithms (AFN and AFN-CMA-ES)
  python3 run_afn.py --algorithms all --functions 1,2,3 --dimensions 2,5 --max_evals 200
  
  # Test on higher dimensions with more evaluations
  python3 run_afn.py --algorithms AFN,AFN-CMA-ES --functions 1-5 --dimensions 10 --n_runs 30 --max_evals 1000 --verbose
        """
    )
    
    parser.add_argument('--algorithms', type=str, default='AFN,AFN-CMA-ES',
                       help='Algorithms to run (comma-separated). Options: AFN, AFN-CMA-ES, or "all" (default: AFN,AFN-CMA-ES)')
    parser.add_argument('--functions', type=str, default='1,2,3',
                       help='Function IDs (comma-separated or range, e.g., "1,2,3" or "1-24") (default: 1,2,3)')
    parser.add_argument('--dimensions', type=str, default='2,5',
                       help='Dimensions (comma-separated, e.g., "2,5,10") (default: 2,5)')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs per test case (default: 10)')
    parser.add_argument('--max_evals', type=int, default=200,
                       help='Maximum function evaluations per run (default: 200)')
    parser.add_argument('--model_type', type=str, default='mlp',
                       choices=['random_forest', 'mlp'],
                       help='Surrogate model type for AFN (random_forest or mlp) (default: mlp)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress during optimization')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Parse lists
    test_functions = parse_list_arg(args.functions)
    dimensions = parse_list_arg(args.dimensions)
    
    # Parse algorithms
    if args.algorithms.lower() == 'all':
        algorithms = ["AFN", "AFN-CMA-ES"]
    else:
        algorithms = [alg.strip() for alg in args.algorithms.split(',')]
    
    # Print configuration
    print("\n" + "="*80)
    print("AFN vs CMA-ES VARIANTS COMPARISON")
    print("="*80)
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Surrogate Model: {args.model_type}")
    print(f"Test Functions: {test_functions}")
    print(f"Dimensions: {dimensions}")
    print(f"Runs per test: {args.n_runs}")
    print(f"Max evaluations: {args.max_evals}")
    print(f"Output directory: {args.output_dir}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print("="*80)
    
    # Create comparison object
    comparison = OptimizationComparison(
        test_functions=test_functions,
        dimensions=dimensions,
        n_runs=args.n_runs,
        max_evaluations=args.max_evals,
        save_dir=args.output_dir,
        algorithms=algorithms,
        model_type=args.model_type
    )
    
    # Run comparison
    comparison.run(verbose=args.verbose)
    
    # Minimal success message
    print("\n" + "="*80)
    print("✓ Comparison completed successfully!")
    print("="*80)
    print(f"\nResults saved to: {comparison.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
