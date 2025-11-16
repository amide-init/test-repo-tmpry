"""
AFN Core using ensemble regressors for surrogate modeling.
This module implements a surrogate ensemble using scikit-learn regressors
with uncertainty estimated via ensemble variance. The optimization loop and
public interface mirror the original AFNCore so it can be drop-in used.

Supports two model types:
- 'random_forest': Random Forest ensemble (default, fast, sample-efficient)
- 'mlp': Multi-Layer Perceptron ensemble (deep learning, more capacity)
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
import warnings

# Lightweight sklearn imports (avoid heavy deps if not installed)
try:
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.neural_network import MLPRegressor
	from sklearn.model_selection import train_test_split
	sk_ok = True
except Exception:
	sk_ok = False
	RandomForestRegressor = object  # type: ignore
	MLPRegressor = object  # type: ignore
	def train_test_split(*args, **kwargs):  # type: ignore
		return args


class EnsembleRegressor:
	"""Simple ensemble wrapper around scikit-learn regressors.
	
	Supports two model types:
	- 'random_forest': Fast, sample-efficient, good for few samples
	- 'mlp': Neural network, more capacity, needs more samples
	"""
	def __init__(self, input_dim: int, n_models: int = 5, model_type: str = 'random_forest', random_state: int = 42):
		self.input_dim = input_dim
		self.n_models = n_models
		self.model_type = model_type.lower()
		self.models = []
		self.random_state = random_state
		
		if not sk_ok:
			warnings.warn("scikit-learn not available. Models will not be created.")
			return
		
		# Validate model type
		if self.model_type not in ['random_forest', 'mlp']:
			raise ValueError(f"model_type must be 'random_forest' or 'mlp', got '{self.model_type}'")
		
		# Create ensemble models
		for i in range(n_models):
			if self.model_type == 'random_forest':
				model = RandomForestRegressor(
					n_estimators=100,
					max_depth=None,
					min_samples_split=2,
					min_samples_leaf=1,
					random_state=self.random_state + i,
					n_jobs=-1,
				)
			elif self.model_type == 'mlp':
				model = MLPRegressor(
					hidden_layer_sizes=(64, 32, 16),
					activation='relu',
					solver='adam',
					alpha=0.001,  # L2 regularization
					batch_size='auto',
					learning_rate='adaptive',
					learning_rate_init=0.001,
					max_iter=300,
					early_stopping=True,
					validation_fraction=0.1,
					n_iter_no_change=10,
					random_state=self.random_state + i,
					warm_start=False,
				)
			
			self.models.append(model)

	def fit(self, X: np.ndarray, y: np.ndarray) -> None:
		if not self.models:
			return
		
		# For MLP, need more samples to avoid convergence issues
		min_samples = 20 if self.model_type == 'mlp' else 5
		if len(X) < min_samples:
			warnings.warn(f"Not enough samples ({len(X)}) for {self.model_type}. Need at least {min_samples}.")
			return
		
		# Suppress convergence warnings for MLP during training
		with warnings.catch_warnings():
			if self.model_type == 'mlp':
				warnings.filterwarnings('ignore', category=UserWarning)
			
			for model in self.models:
				try:
					model.fit(X, y)
				except Exception as e:
					warnings.warn(f"Model fitting failed: {e}")

	def predict(self, X: np.ndarray) -> np.ndarray:
		if not self.models:
			return np.zeros(len(X))
		preds = [m.predict(X) for m in self.models]
		return np.mean(np.stack(preds, axis=0), axis=0)

	def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if not self.models:
			return np.zeros(len(X)), np.zeros(len(X))
		preds = [m.predict(X) for m in self.models]
		preds = np.stack(preds, axis=0)  # (n_models, N)
		mean_pred = np.mean(preds, axis=0)
		std_pred = np.std(preds, axis=0)
		return mean_pred, std_pred


class AFNCore:
	"""
	AFN Core using ensemble regressors as surrogate.
	Public API mirrors the original AFNCore:
	- constructor params
	- optimize(objective_function, verbose=True) returns dict with keys:
	  best_x, best_y, X_history, y_history, history (alias), evaluation_count
	
	Args:
		input_dim: Dimension of input space
		bounds: Search space bounds [(low, high), ...]
		uncertainty_threshold: Threshold for high uncertainty exploration
		batch_size: Number of points to evaluate per iteration
		max_evaluations: Maximum function evaluations
		convergence_threshold: Convergence threshold for improvement
		convergence_window: Window size for convergence check
		n_models: Number of models in ensemble
		model_type: Type of surrogate model ('random_forest' or 'mlp')
		random_state: Random seed
	"""
	def __init__(
		self,
		input_dim: int,
		bounds: List[Tuple[float, float]],
		uncertainty_threshold: float = 0.03,
		batch_size: int = 8,
		max_evaluations: int = 100,
		convergence_threshold: float = 1e-6,
		convergence_window: int = 10,
		n_models: int = 5,
		model_type: str = 'random_forest',
		random_state: int = 42,
	):
		self.input_dim = input_dim
		self.bounds = np.array(bounds, dtype=float)
		self.uncertainty_threshold = uncertainty_threshold
		self.batch_size = batch_size
		self.max_evaluations = max_evaluations
		self.convergence_threshold = convergence_threshold
		self.convergence_window = convergence_window
		self.model_type = model_type
		self.random_state = random_state
		self.rng = np.random.RandomState(random_state)

		self.ensemble = EnsembleRegressor(
			input_dim, 
			n_models=n_models, 
			model_type=model_type,
			random_state=random_state
		)

		self.X_history: List[np.ndarray] = []
		self.y_history: List[float] = []
		self.best_x: Optional[np.ndarray] = None
		self.best_y: float = float('inf')
		self.improvement_history: List[float] = []
		self.evaluation_count: int = 0

	def generate_initial_samples(self, n_samples: int = None) -> np.ndarray:
		if n_samples is None:
			n_samples = max(30, min(200, self.input_dim * 10))
		samples = self.rng.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(n_samples, self.input_dim))
		return samples

	def update_history(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
		if X_new.ndim == 1:
			X_new = X_new.reshape(1, -1)
		y_new = np.asarray(y_new).flatten()
		self.X_history.extend(list(X_new))
		self.y_history.extend(list(y_new))
		# Update best
		idx = int(np.argmin(self.y_history)) if len(self.y_history) > 0 else None
		if idx is not None:
			self.best_y = float(self.y_history[idx])
			self.best_x = np.array(self.X_history[idx])
		# Track improvement
		if len(self.y_history) >= 2:
			imp = abs(self.y_history[-2] - self.y_history[-1])
			self.improvement_history.append(imp)

	def fit_surrogate(self) -> None:
		if len(self.X_history) < 5:
			return
		X = np.array(self.X_history, dtype=float)
		y = np.array(self.y_history, dtype=float)
		self.ensemble.fit(X, y)

	def select_candidate_points(self, n_candidates: int = 1000) -> np.ndarray:
		# Sample candidate grid uniformly
		cands = self.rng.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(n_candidates, self.input_dim))
		if len(self.X_history) >= 5:
			mean_pred, std_pred = self.ensemble.predict_with_uncertainty(cands)
			# Potential optima (lowest predictions)
			opt_idx = np.argsort(mean_pred)[: max(1, self.batch_size)]
			# High uncertainty
			unc_idx = np.where(std_pred > self.uncertainty_threshold)[0]
			# Combine and rank by uncertainty + inverse predicted value
			sel = np.unique(np.concatenate([opt_idx, unc_idx]))
			if sel.size == 0:
				return cands[opt_idx]
			scores = std_pred[sel] + 1.0 / (1e-9 + np.maximum(mean_pred[sel], 1e-9))
			best = sel[np.argsort(scores)[-self.batch_size:]]
			return cands[best]
		return cands[: self.batch_size]

	def local_search_around_best(self, n_points: int = 10) -> np.ndarray:
		if self.best_x is None:
			return np.empty((0, self.input_dim))
		scale = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
		pts = []
		for _ in range(n_points):
			noise = self.rng.normal(0.0, scale, size=self.input_dim)
			p = np.clip(self.best_x + noise, self.bounds[:, 0], self.bounds[:, 1])
			pts.append(p)
		return np.array(pts)

	def optimize(self, objective_function: Callable, verbose: bool = True):
		if verbose:
			print(f"AFN optimization start (surrogate model: {self.model_type})")
		# Initial design
		X0 = self.generate_initial_samples()
		y0 = np.array([objective_function(x) for x in X0], dtype=float)
		self.update_history(X0, y0)
		self.evaluation_count += len(y0)
		if verbose:
			print(f"  Initial best: {self.best_y:.6f}")

		while self.evaluation_count < self.max_evaluations:
			# Fit surrogate
			self.fit_surrogate()
			# Determine current batch size (simple adaptive rule)
			current_batch = self.batch_size
			if len(self.improvement_history) > 3:
				recent = float(np.mean(self.improvement_history[-3:]))
				if recent < 1e-3:
					current_batch = min(self.batch_size * 2, 16)
				elif recent > 1e-2:
					current_batch = max(self.batch_size // 2, 2)
			# Select candidates (hybrid)
			if len(self.X_history) >= 30:
				cand = self.select_candidate_points(n_candidates=2000)
				local = self.local_search_around_best(n_points=current_batch)
				cand = np.vstack([cand, local])[:current_batch]
			elif len(self.X_history) >= 10:
				cand = self.select_candidate_points(n_candidates=1500)[:current_batch]
			else:
				rand = self.generate_initial_samples(n_samples=max(current_batch // 2, 2))
				local = self.local_search_around_best(n_points=max(current_batch - len(rand), 0))
				cand = np.vstack([rand, local]) if local.size else rand
			# Evaluate
			y_new = np.array([objective_function(x) for x in cand], dtype=float)
			self.update_history(cand, y_new)
			self.evaluation_count += len(y_new)
			if verbose:
				print(f"  Eval {self.evaluation_count}: best={self.best_y:.6f}")
			# Convergence check
			if len(self.improvement_history) >= self.convergence_window:
				window = self.improvement_history[-self.convergence_window:]
				if max(window) < self.convergence_threshold:
					if verbose:
						print("  Converged by improvement window")
					break

		# Return in the same shape as original AFNCore
		return {
			'best_x': self.best_x,
			'best_y': self.best_y,
			'X_history': np.array(self.X_history, dtype=float),
			'y_history': np.array(self.y_history, dtype=float),
			'history': np.array(self.y_history, dtype=float),  # alias for compatibility
			'evaluation_count': self.evaluation_count,
			'converged': self.evaluation_count < self.max_evaluations,  # True if converged early
		}
