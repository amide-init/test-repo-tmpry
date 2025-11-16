#!/usr/bin/env python3
"""
CMA-ES variants for comparison with AFN
Implements:
- CMA-ES: Standard Covariance Matrix Adaptation Evolution Strategy
- AFN-CMA-ES: CMA-ES with AFN surrogate model (Random Forest ensemble)
- LQ-CMA-ES: CMA-ES with Linear-Quadratic surrogate (Hansen et al.)
- DTS-CMA-ES: CMA-ES with Dynamic Threshold Selection (Bajer et al.)
- LMM-CMA-ES: CMA-ES with Local Meta-Model (Loshchilov et al.)
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import cma for standard CMA-ES
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("Warning: cma not available. Install with: pip install cma")

# Import sklearn for surrogate models
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class CMAEvolutionStrategy:
    """Standard CMA-ES implementation using pycma library"""
    
    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 max_evaluations: int = 100,
                 population_size: Optional[int] = None,
                 sigma0: Optional[float] = None,
                 random_state: int = 42):
        
        self.bounds = np.array(bounds, dtype=float)
        self.dimension = len(bounds)
        self.max_evaluations = max_evaluations
        self.population_size = population_size if population_size else max(6, 4 + int(3 * np.log(self.dimension)))
        self.random_state = random_state
        
        # Initial mean and sigma
        self.x0 = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        self.sigma0 = sigma0 if sigma0 else np.mean((self.bounds[:, 1] - self.bounds[:, 0]) / 6.0)
        
        # Results tracking
        self.best_x = None
        self.best_y = float('inf')
        self.y_history = []
        self.evaluation_count = 0
        
    def optimize(self, objective_function: Callable, verbose: bool = True):
        """Run standard CMA-ES optimization"""
        if not CMA_AVAILABLE:
            raise ImportError("cma library not available")
        
        if verbose:
            print("Running Standard CMA-ES...")
        
        opts = {
            'bounds': [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
            'popsize': self.population_size,
            'maxfevals': self.max_evaluations,
            'seed': self.random_state,
            'verbose': -1 if not verbose else 0,
        }
        
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, opts)
        
        while not es.stop() and self.evaluation_count < self.max_evaluations:
            solutions = es.ask()
            fitness_values = [objective_function(np.array(x)) for x in solutions]
            es.tell(solutions, fitness_values)
            
            # Update history
            for f in fitness_values:
                self.y_history.append(f)
                if f < self.best_y:
                    self.best_y = f
            
            self.evaluation_count += len(solutions)
            
            if verbose and self.evaluation_count % 50 == 0:
                print(f"  Evals: {self.evaluation_count}, Best: {self.best_y:.6f}")
        
        self.best_x = es.result.xbest
        self.best_y = es.result.fbest
        
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'y_history': np.array(self.y_history),
            'history': np.array(self.y_history),
            'evaluation_count': self.evaluation_count,
            'converged': self.evaluation_count < self.max_evaluations
        }


class AFN_CMA:
    """AFN-CMA-ES: CMA-ES with AFN Ensemble Surrogate (Random Forest or MLP)"""
    
    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 max_evaluations: int = 100,
                 population_size: Optional[int] = None,
                 sigma0: Optional[float] = None,
                 n_models: int = 5,
                 surrogate_interval: int = 5,
                 model_type: str = 'random_forest',
                 random_state: int = 42):
        
        self.bounds = np.array(bounds, dtype=float)
        self.dimension = len(bounds)
        self.max_evaluations = max_evaluations
        self.population_size = population_size if population_size else max(6, 4 + int(3 * np.log(self.dimension)))
        self.n_models = n_models
        self.surrogate_interval = surrogate_interval  # Use surrogate every N generations
        self.model_type = model_type.lower()
        self.random_state = random_state
        
        # Initial parameters
        self.x0 = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        self.sigma0 = sigma0 if sigma0 else np.mean((self.bounds[:, 1] - self.bounds[:, 0]) / 6.0)
        
        # Surrogate models (Random Forest or MLP ensemble like AFN)
        self.ensemble = []
        if SKLEARN_AVAILABLE:
            for i in range(n_models):
                if self.model_type == 'mlp':
                    from sklearn.neural_network import MLPRegressor
                    model = MLPRegressor(
                        hidden_layer_sizes=(64, 32, 16),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=random_state + i,
                        warm_start=False
                    )
                else:  # random_forest (default)
                    model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        random_state=random_state + i,
                        n_jobs=-1
                    )
                self.ensemble.append(model)
        
        # Results tracking
        self.best_x = None
        self.best_y = float('inf')
        self.y_history = []
        self.X_archive = []
        self.y_archive = []
        self.evaluation_count = 0
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble and estimate uncertainty"""
        if not self.ensemble or len(self.X_archive) < 10:
            return np.zeros(len(X)), np.ones(len(X)) * 1e10
        
        try:
            predictions = [model.predict(X) for model in self.ensemble]
            predictions = np.stack(predictions, axis=0)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            return mean_pred, std_pred
        except:
            return np.zeros(len(X)), np.ones(len(X)) * 1e10
    
    def fit_surrogate(self):
        """Fit the ensemble surrogate models"""
        if len(self.X_archive) < 10 or not SKLEARN_AVAILABLE:
            return
        
        X = np.array(self.X_archive)
        y = np.array(self.y_archive)
        
        for model in self.ensemble:
            try:
                model.fit(X, y)
            except:
                pass
    
    def optimize(self, objective_function: Callable, verbose: bool = True):
        """Run AFN-CMA-ES optimization"""
        if not CMA_AVAILABLE:
            raise ImportError("cma library not available")
        
        if verbose:
            surrogate_name = "MLP Deep Ensemble" if self.model_type == 'mlp' else "Random Forest Ensemble"
            print(f"Running AFN-CMA-ES (CMA-ES + {surrogate_name} Surrogate)...")
        
        opts = {
            'bounds': [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
            'popsize': self.population_size,
            'maxfevals': self.max_evaluations,
            'seed': self.random_state,
            'verbose': -1,
        }
        
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, opts)
        generation = 0
        
        while not es.stop() and self.evaluation_count < self.max_evaluations:
            solutions = es.ask()
            
            # Every N generations, use surrogate for pre-selection
            use_surrogate = (generation % self.surrogate_interval == 0) and len(self.X_archive) >= 10
            
            if use_surrogate and SKLEARN_AVAILABLE:
                self.fit_surrogate()
                X_candidates = np.array(solutions)
                mean_pred, std_pred = self.predict_with_uncertainty(X_candidates)
                
                # Rank by uncertainty + predicted fitness (exploration-exploitation)
                scores = -mean_pred + 0.5 * std_pred  # Higher score = better
                top_indices = np.argsort(scores)[-self.population_size:]
                solutions = [solutions[i] for i in top_indices]
            
            # Evaluate with real objective function
            fitness_values = []
            for x in solutions:
                f = objective_function(np.array(x))
                fitness_values.append(f)
                
                # Archive for surrogate training
                self.X_archive.append(x)
                self.y_archive.append(f)
                self.y_history.append(f)
                
                if f < self.best_y:
                    self.best_y = f
                    self.best_x = np.array(x)
                
                self.evaluation_count += 1
                if self.evaluation_count >= self.max_evaluations:
                    break
            
            es.tell(solutions[:len(fitness_values)], fitness_values)
            generation += 1
            
            if verbose and self.evaluation_count % 50 == 0:
                print(f"  Evals: {self.evaluation_count}, Best: {self.best_y:.6f}")
        
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'y_history': np.array(self.y_history),
            'history': np.array(self.y_history),
            'evaluation_count': self.evaluation_count,
            'converged': self.evaluation_count < self.max_evaluations
        }


class LQ_CMA:
    """LQ-CMA-ES: CMA-ES with Linear-Quadratic surrogate (Hansen et al., 2019)"""
    
    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 max_evaluations: int = 100,
                 population_size: Optional[int] = None,
                 sigma0: Optional[float] = None,
                 surrogate_interval: int = 5,
                 random_state: int = 42):
        
        self.bounds = np.array(bounds, dtype=float)
        self.dimension = len(bounds)
        self.max_evaluations = max_evaluations
        self.population_size = population_size if population_size else max(6, 4 + int(3 * np.log(self.dimension)))
        self.surrogate_interval = surrogate_interval
        self.random_state = random_state
        
        # Initial parameters
        self.x0 = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        self.sigma0 = sigma0 if sigma0 else np.mean((self.bounds[:, 1] - self.bounds[:, 0]) / 6.0)
        
        # Linear-Quadratic surrogate
        self.poly_features = None
        self.lq_model = None
        if SKLEARN_AVAILABLE:
            self.poly_features = PolynomialFeatures(degree=2, include_bias=True)
            self.lq_model = Ridge(alpha=1.0)
        
        # Results tracking
        self.best_x = None
        self.best_y = float('inf')
        self.y_history = []
        self.X_archive = []
        self.y_archive = []
        self.evaluation_count = 0
        
    def fit_surrogate(self):
        """Fit linear-quadratic surrogate model"""
        if len(self.X_archive) < 15 or not SKLEARN_AVAILABLE:
            return False
        
        try:
            X = np.array(self.X_archive)
            y = np.array(self.y_archive)
            
            # Transform to polynomial features
            X_poly = self.poly_features.fit_transform(X)
            self.lq_model.fit(X_poly, y)
            return True
        except:
            return False
    
    def predict_lq(self, X: np.ndarray) -> np.ndarray:
        """Predict using linear-quadratic model"""
        if self.lq_model is None or len(self.X_archive) < 15:
            return np.zeros(len(X))
        
        try:
            X_poly = self.poly_features.transform(X)
            return self.lq_model.predict(X_poly)
        except:
            return np.zeros(len(X))
    
    def optimize(self, objective_function: Callable, verbose: bool = True):
        """Run LQ-CMA-ES optimization"""
        if not CMA_AVAILABLE:
            raise ImportError("cma library not available")
        
        if verbose:
            print("Running LQ-CMA-ES (CMA-ES + Linear-Quadratic Surrogate)...")
        
        opts = {
            'bounds': [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
            'popsize': self.population_size,
            'maxfevals': self.max_evaluations,
            'seed': self.random_state,
            'verbose': -1,
        }
        
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, opts)
        generation = 0
        
        while not es.stop() and self.evaluation_count < self.max_evaluations:
            solutions = es.ask()
            
            # Use surrogate for pre-screening
            use_surrogate = (generation % self.surrogate_interval == 0) and len(self.X_archive) >= 15
            
            if use_surrogate and SKLEARN_AVAILABLE and self.fit_surrogate():
                X_candidates = np.array(solutions)
                lq_predictions = self.predict_lq(X_candidates)
                
                # Select best according to LQ model (lower is better)
                top_indices = np.argsort(lq_predictions)[:self.population_size]
                solutions = [solutions[i] for i in top_indices]
            
            # Evaluate with real objective function
            fitness_values = []
            for x in solutions:
                f = objective_function(np.array(x))
                fitness_values.append(f)
                
                # Archive
                self.X_archive.append(x)
                self.y_archive.append(f)
                self.y_history.append(f)
                
                if f < self.best_y:
                    self.best_y = f
                    self.best_x = np.array(x)
                
                self.evaluation_count += 1
                if self.evaluation_count >= self.max_evaluations:
                    break
            
            es.tell(solutions[:len(fitness_values)], fitness_values)
            generation += 1
            
            if verbose and self.evaluation_count % 50 == 0:
                print(f"  Evals: {self.evaluation_count}, Best: {self.best_y:.6f}")
        
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'y_history': np.array(self.y_history),
            'history': np.array(self.y_history),
            'evaluation_count': self.evaluation_count,
            'converged': self.evaluation_count < self.max_evaluations
        }


class DTS_CMA:
    """DTS-CMA-ES: CMA-ES with Dynamic Threshold Selection (Bajer et al., 2019)
    Uses Gaussian Process with Expected Improvement for dynamic threshold"""
    
    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 max_evaluations: int = 100,
                 population_size: Optional[int] = None,
                 sigma0: Optional[float] = None,
                 surrogate_interval: int = 5,
                 random_state: int = 42):
        
        self.bounds = np.array(bounds, dtype=float)
        self.dimension = len(bounds)
        self.max_evaluations = max_evaluations
        self.population_size = population_size if population_size else max(6, 4 + int(3 * np.log(self.dimension)))
        self.surrogate_interval = surrogate_interval
        self.random_state = random_state
        
        # Initial parameters
        self.x0 = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        self.sigma0 = sigma0 if sigma0 else np.mean((self.bounds[:, 1] - self.bounds[:, 0]) / 6.0)
        
        # GP surrogate for DTS
        self.gp_model = None
        if SKLEARN_AVAILABLE:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                random_state=random_state,
                alpha=1e-6
            )
        
        # Results tracking
        self.best_x = None
        self.best_y = float('inf')
        self.y_history = []
        self.X_archive = []
        self.y_archive = []
        self.evaluation_count = 0
        
    def fit_gp_surrogate(self):
        """Fit GP surrogate model"""
        if len(self.X_archive) < 10 or not SKLEARN_AVAILABLE:
            return False
        
        try:
            X = np.array(self.X_archive)
            y = np.array(self.y_archive)
            self.gp_model.fit(X, y)
            return True
        except:
            return False
    
    def expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Calculate Expected Improvement acquisition function"""
        if self.gp_model is None or len(self.X_archive) < 10:
            return np.zeros(len(X))
        
        try:
            mu, sigma = self.gp_model.predict(X, return_std=True)
            mu = mu.flatten()
            sigma = sigma.flatten()
            
            # Current best
            f_best = self.best_y
            
            # Calculate EI
            with np.errstate(divide='warn'):
                Z = (f_best - mu) / (sigma + 1e-9)
                ei = (f_best - mu) * norm_cdf(Z) + sigma * norm_pdf(Z)
                ei[sigma == 0.0] = 0.0
            
            return ei
        except:
            return np.zeros(len(X))
    
    def optimize(self, objective_function: Callable, verbose: bool = True):
        """Run DTS-CMA-ES optimization"""
        if not CMA_AVAILABLE:
            raise ImportError("cma library not available")
        
        if verbose:
            print("Running DTS-CMA-ES (CMA-ES + GP with Dynamic Threshold)...")
        
        opts = {
            'bounds': [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
            'popsize': self.population_size,
            'maxfevals': self.max_evaluations,
            'seed': self.random_state,
            'verbose': -1,
        }
        
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, opts)
        generation = 0
        
        while not es.stop() and self.evaluation_count < self.max_evaluations:
            solutions = es.ask()
            
            # Use GP surrogate with EI for dynamic threshold selection
            use_surrogate = (generation % self.surrogate_interval == 0) and len(self.X_archive) >= 10
            
            if use_surrogate and SKLEARN_AVAILABLE and self.fit_gp_surrogate():
                X_candidates = np.array(solutions)
                ei_values = self.expected_improvement(X_candidates)
                
                # Select candidates with highest EI
                top_indices = np.argsort(ei_values)[-self.population_size:]
                solutions = [solutions[i] for i in top_indices]
            
            # Evaluate with real objective function
            fitness_values = []
            for x in solutions:
                f = objective_function(np.array(x))
                fitness_values.append(f)
                
                # Archive
                self.X_archive.append(x)
                self.y_archive.append(f)
                self.y_history.append(f)
                
                if f < self.best_y:
                    self.best_y = f
                    self.best_x = np.array(x)
                
                self.evaluation_count += 1
                if self.evaluation_count >= self.max_evaluations:
                    break
            
            es.tell(solutions[:len(fitness_values)], fitness_values)
            generation += 1
            
            if verbose and self.evaluation_count % 50 == 0:
                print(f"  Evals: {self.evaluation_count}, Best: {self.best_y:.6f}")
        
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'y_history': np.array(self.y_history),
            'history': np.array(self.y_history),
            'evaluation_count': self.evaluation_count,
            'converged': self.evaluation_count < self.max_evaluations
        }


class LMM_CMA:
    """LMM-CMA-ES: CMA-ES with Local Meta-Model (Loshchilov et al., 2012)
    Uses local GP models around promising regions"""
    
    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 max_evaluations: int = 100,
                 population_size: Optional[int] = None,
                 sigma0: Optional[float] = None,
                 surrogate_interval: int = 5,
                 local_size: int = 50,
                 random_state: int = 42):
        
        self.bounds = np.array(bounds, dtype=float)
        self.dimension = len(bounds)
        self.max_evaluations = max_evaluations
        self.population_size = population_size if population_size else max(6, 4 + int(3 * np.log(self.dimension)))
        self.surrogate_interval = surrogate_interval
        self.local_size = local_size  # Size of local training set
        self.random_state = random_state
        
        # Initial parameters
        self.x0 = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        self.sigma0 = sigma0 if sigma0 else np.mean((self.bounds[:, 1] - self.bounds[:, 0]) / 6.0)
        
        # Local meta-model (GP)
        self.local_gp = None
        if SKLEARN_AVAILABLE:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            self.local_gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=3,
                random_state=random_state,
                alpha=1e-6
            )
        
        # Results tracking
        self.best_x = None
        self.best_y = float('inf')
        self.y_history = []
        self.X_archive = []
        self.y_archive = []
        self.evaluation_count = 0
        
    def get_local_dataset(self, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get local dataset around a center point"""
        if len(self.X_archive) < 10:
            return None, None
        
        X = np.array(self.X_archive)
        y = np.array(self.y_archive)
        
        # Calculate distances from center
        distances = np.linalg.norm(X - center, axis=1)
        
        # Select closest points
        n_local = min(self.local_size, len(X))
        local_indices = np.argsort(distances)[:n_local]
        
        return X[local_indices], y[local_indices]
    
    def fit_local_surrogate(self, center: np.ndarray):
        """Fit local GP model around center"""
        if not SKLEARN_AVAILABLE:
            return False
        
        X_local, y_local = self.get_local_dataset(center)
        
        if X_local is None or len(X_local) < 5:
            return False
        
        try:
            self.local_gp.fit(X_local, y_local)
            return True
        except:
            return False
    
    def predict_local(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using local GP model"""
        if self.local_gp is None:
            return np.zeros(len(X)), np.ones(len(X))
        
        try:
            mu, sigma = self.local_gp.predict(X, return_std=True)
            return mu.flatten(), sigma.flatten()
        except:
            return np.zeros(len(X)), np.ones(len(X))
    
    def optimize(self, objective_function: Callable, verbose: bool = True):
        """Run LMM-CMA-ES optimization"""
        if not CMA_AVAILABLE:
            raise ImportError("cma library not available")
        
        if verbose:
            print("Running LMM-CMA-ES (CMA-ES + Local Meta-Model)...")
        
        opts = {
            'bounds': [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
            'popsize': self.population_size,
            'maxfevals': self.max_evaluations,
            'seed': self.random_state,
            'verbose': -1,
        }
        
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, opts)
        generation = 0
        
        while not es.stop() and self.evaluation_count < self.max_evaluations:
            solutions = es.ask()
            
            # Use local surrogate around current mean
            use_surrogate = (generation % self.surrogate_interval == 0) and len(self.X_archive) >= 10
            
            if use_surrogate and SKLEARN_AVAILABLE:
                current_mean = np.array(es.mean)
                
                if self.fit_local_surrogate(current_mean):
                    X_candidates = np.array(solutions)
                    mu_pred, sigma_pred = self.predict_local(X_candidates)
                    
                    # Lower Confidence Bound acquisition
                    lcb = mu_pred - 2.0 * sigma_pred
                    
                    # Select candidates with best LCB
                    top_indices = np.argsort(lcb)[:self.population_size]
                    solutions = [solutions[i] for i in top_indices]
            
            # Evaluate with real objective function
            fitness_values = []
            for x in solutions:
                f = objective_function(np.array(x))
                fitness_values.append(f)
                
                # Archive
                self.X_archive.append(x)
                self.y_archive.append(f)
                self.y_history.append(f)
                
                if f < self.best_y:
                    self.best_y = f
                    self.best_x = np.array(x)
                
                self.evaluation_count += 1
                if self.evaluation_count >= self.max_evaluations:
                    break
            
            es.tell(solutions[:len(fitness_values)], fitness_values)
            generation += 1
            
            if verbose and self.evaluation_count % 50 == 0:
                print(f"  Evals: {self.evaluation_count}, Best: {self.best_y:.6f}")
        
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'y_history': np.array(self.y_history),
            'history': np.array(self.y_history),
            'evaluation_count': self.evaluation_count,
            'converged': self.evaluation_count < self.max_evaluations
        }


# Helper functions for DTS-CMA (normal distribution)
def norm_pdf(x):
    """Standard normal PDF"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def norm_cdf(x):
    """Standard normal CDF"""
    from scipy.special import erf
    return 0.5 * (1 + erf(x / np.sqrt(2)))


# Test function
def test_cmaes_variants():
    """Test all CMA-ES variants on simple sphere function"""
    
    def sphere(x):
        return np.sum(x**2)
    
    bounds = [(-5.0, 5.0)] * 5
    max_evals = 200
    
    optimizers = {
        "CMA-ES": CMAEvolutionStrategy,
        "AFN-CMA-ES": AFN_CMA,
        "LQ-CMA-ES": LQ_CMA,
        "DTS-CMA-ES": DTS_CMA,
        "LMM-CMA-ES": LMM_CMA
    }
    
    print("Testing CMA-ES variants on 5D Sphere function...")
    print("=" * 60)
    
    results = {}
    for name, OptimizerClass in optimizers.items():
        print(f"\n{name}:")
        try:
            optimizer = OptimizerClass(
                bounds=bounds,
                max_evaluations=max_evals,
                random_state=42
            )
            result = optimizer.optimize(sphere, verbose=False)
            results[name] = result
            print(f"  Best value: {result['best_y']:.6e}")
            print(f"  Evaluations: {result['evaluation_count']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


if __name__ == "__main__":
    test_cmaes_variants()

