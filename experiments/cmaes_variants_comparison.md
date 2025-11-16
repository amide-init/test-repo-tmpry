# CMA-ES Variants Comparison

This document describes the comparison of different CMA-ES (Covariance Matrix Adaptation Evolution Strategy) variants with surrogate modeling approaches.

## Optimizers Compared

### 1. **CMA-ES** (Standard)
- **Description**: Standard Covariance Matrix Adaptation Evolution Strategy
- **Reference**: Hansen & Ostermeier (2001)
- **Characteristics**: 
  - Population-based evolutionary strategy
  - Adaptive step-size control
  - Covariance matrix adaptation for search distribution
  - No surrogate model

### 2. **AFN-CMA-ES** (Adaptive Function Nexus)
- **Description**: CMA-ES enhanced with Random Forest ensemble surrogate
- **Our Novel Approach**
- **Characteristics**:
  - Ensemble of 5 Random Forest regressors (50 trees each)
  - Uncertainty estimation via ensemble variance
  - Pre-selection using surrogate predictions every N generations
  - Exploration-exploitation balance: score = -mean_pred + 0.5 * std_pred

### 3. **LQ-CMA-ES** (Linear-Quadratic)
- **Description**: CMA-ES with linear-quadratic surrogate model
- **Reference**: Hansen et al. (2019)
- **Characteristics**:
  - Global polynomial approximation (degree 2)
  - Ridge regression for regularization
  - Pre-screening candidates based on LQ model predictions
  - Simple and computationally efficient

### 4. **DTS-CMA-ES** (Dynamic Threshold Selection)
- **Description**: CMA-ES with Gaussian Process and Expected Improvement
- **Reference**: Bajer et al. (2019)
- **Characteristics**:
  - Gaussian Process (GP) surrogate with RBF kernel
  - Expected Improvement (EI) acquisition function
  - Dynamic threshold based on GP uncertainty
  - Balances exploration and exploitation

### 5. **LMM-CMA-ES** (Local Meta-Model)
- **Description**: CMA-ES with local Gaussian Process models
- **Reference**: Loshchilov et al. (2012)
- **Characteristics**:
  - Local GP models around promising regions
  - Adaptive local dataset selection (50 nearest points)
  - Lower Confidence Bound (LCB) acquisition: LCB = μ - 2σ
  - Focuses computational effort on local optimization

## Evaluation Metrics

### 1. **Optimization Accuracy** (%)
- Measures how close the algorithm gets to the known optimum
- Formula: `100 × (1 - (best_found - optimum) / (initial_value - optimum))`
- Higher is better

### 2. **Convergence Speed** (%)
- Percentage of budget used to reach 95% of final improvement
- Formula: Evaluations to reach 95% improvement / max_evaluations × 100
- Lower is better (faster convergence)

### 3. **Resource Utilization** (%)
- Percentage of evaluation budget actually used
- Formula: `evaluations_used / max_evaluations × 100`
- Shows algorithm efficiency

### 4. **Robustness** (%)
- Consistency across multiple runs
- Formula: `100 × (1 - std(best_values) / mean(best_values))`
- Higher is better (more consistent)

## Benchmark Functions

Using **COCO/BBOB** (Black-Box Optimization Benchmarking) suite:

### Test Functions
- **Function 1**: Sphere (unimodal, separable)
- **Function 2**: Ellipsoid (unimodal, moderate conditioning)
- **Function 3**: Rastrigin (multimodal, separable)

### Dimensions
- **2D**: Low-dimensional (visualization possible)
- **5D**: Medium-dimensional (common benchmark)
- **10D**: Higher-dimensional (more challenging)

## Running the Comparison

### Basic Usage
```bash
python run_cmaes_comparison.py
```

### Custom Settings
```bash
python run_cmaes_comparison.py \
    --functions 1,2,3,8,10 \
    --dimensions 2,5,10 \
    --n_runs 20 \
    --max_evals 500 \
    --verbose
```

### Arguments
- `--functions`: Comma-separated BBOB function IDs (default: 1,2,3)
- `--dimensions`: Comma-separated dimensions (default: 2,5)
- `--n_runs`: Number of independent runs per test (default: 10)
- `--max_evals`: Maximum function evaluations per run (default: 200)
- `--output_dir`: Results directory (default: results)
- `--verbose`: Show detailed progress

## Expected Outcomes

### Hypothesis

**AFN-CMA-ES** should demonstrate competitive or superior performance because:

1. **Ensemble Uncertainty**: Random Forest ensemble provides robust uncertainty estimates
2. **Exploration-Exploitation Balance**: Explicit trade-off in candidate selection
3. **Scalability**: Random Forests scale better than GPs to higher dimensions
4. **Robustness**: Ensemble averaging reduces overfitting

### Performance Predictions

| Algorithm | Optimization Accuracy | Convergence Speed | Robustness |
|-----------|---------------------|------------------|-----------|
| CMA-ES | Baseline | Baseline | High |
| AFN-CMA-ES | **High** | **Fast** | **High** |
| LQ-CMA-ES | Moderate | Fast | Moderate |
| DTS-CMA-ES | High | Moderate | Moderate |
| LMM-CMA-ES | High | Variable | Moderate |

## Results Location

Results are automatically saved to:
```
results/cmaes_comparison_YYYYMMDD_HHMMSS/
├── config.json              # Experiment configuration
├── runs.json                # Detailed run data
├── metrics_summary.json     # Computed metrics
├── convergence_speed.png    # Convergence comparison
├── optimization_accuracy.png # Accuracy comparison
├── resource_utilization.png  # Resource usage
└── robustness.png           # Consistency comparison
```

## References

1. Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159-195.

2. Hansen, N., Atamna, A., & Auger, A. (2019). How to assess step-size adaptation mechanisms in randomised search. *Parallel Problem Solving from Nature*, 60-73.

3. Bajer, L., Pitra, Z., & Holeňa, M. (2019). Gaussian process surrogate models for the CMA evolution strategy. *Evolutionary Computation*, 27(4), 665-697.

4. Loshchilov, I., Schoenauer, M., & Sebag, M. (2012). Intensive surrogate model exploitation in self-adaptive surrogate-assisted CMA-ES. *Genetic and Evolutionary Computation Conference*, 439-446.

## Testing

Quick test to verify all optimizers work:

```bash
python test_cmaes_variants.py
```

This runs a quick sanity check on a 2D sphere function.

