# 🚀 AFN (Adaptive Fidelity Nexus) Framework

**Paper Implementation**: "Artificial neural networks as surrogate models in optimization"

This repository contains an **ensemble-based implementation** of the AFN algorithm for surrogate-based optimization using COCO/BBOB benchmark functions, with comparisons against CMA-ES variants.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Results & Metrics](#results--metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

The **Adaptive Fidelity Nexus (AFN)** is a novel methodology that adaptively improves an **ensemble regressor** surrogate model by intelligently querying expensive objective functions only where uncertainty is high or where potential optima are found. This leads to more efficient and robust optimization compared to static surrogate models.

### Key Features

✅ **Ensemble Regressor Surrogate** using MLP (neural network)  
✅ **Real COCO/BBOB Benchmark Functions** (24 functions available)  
✅ **Paper-Accurate Implementation** (exact specifications)  
✅ **Uncertainty Quantification** via ensemble variance  
✅ **CMA-ES Integration** with surrogate modeling  
✅ **Lightweight Dependencies** (minimal scikit-learn requirements)  
✅ **Command-Line Interface** for easy usage  
✅ **COCO-Compliant Output** for standardized comparisons

## 🔧 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Virtual Environment Setup (Recommended)

**⚠️ Important**: It's highly recommended to use a virtual environment to avoid dependency conflicts with other Python projects.

#### For Windows (PowerShell/Command Prompt)

```powershell
python -m venv .venv
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For PowerShell:
.\.venv\Scripts\Activate.ps1
# OR for Command Prompt:
.venv\Scripts\activate.bat

# Verify activation (you should see (.venv) in your prompt)
python --version
pip --version
```

#### For macOS/Linux (Terminal)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (you should see (.venv) in your prompt)
python --version
pip --version
```

#### Deactivating Virtual Environment

When you're done working with the project:

```bash
# Deactivate virtual environment (works on all platforms)
deactivate
```

#### Troubleshooting Virtual Environment Issues

**Windows PowerShell Execution Policy Error:**
```powershell
# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again:
.\.venv\Scripts\Activate.ps1
```

**Alternative Windows Activation:**
```cmd
# Use Command Prompt instead of PowerShell
.venv\Scripts\activate.bat
```

**Python3 vs Python Command:**
- On some systems, use `python3` instead of `python`
- Check with: `python --version` or `python3 --version`

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
scikit-learn>=1.1.0
numpy>=1.21.0
matplotlib>=3.5.0
coco-experiment
cocopp
```

## 🚀 Quick Start

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv
```

### Step 2: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Verify activation** (you should see `(.venv)` in your prompt):
```bash
python --version
pip --version
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `scikit-learn>=1.1.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `coco-experiment`
- `cocopp`

### Step 4: Run AFN-CMA-ES Experiments

Run your proposed AFN-CMA-ES algorithm on BBOB benchmark functions:

```bash
python run_afn.py --algorithms AFN-CMA-ES --functions 1-24 --dimensions 2,5,10,20,40 --n_runs 15 --max_evals 5000 --verbose
```

**Command Explanation:**
- `--algorithms AFN-CMA-ES`: Run only AFN-CMA-ES (your proposed algorithm)
- `--functions 1-24`: Test on all 24 BBOB functions
- `--dimensions 2,5,10,20,40`: Test on multiple dimensions
- `--n_runs 15`: Run 15 independent trials per (function, dimension) combination
- `--max_evals 5000`: Maximum function evaluations per run
- `--verbose`: Show detailed progress during optimization

**Note:** The default `model_type` is `mlp` (neural network surrogate) as required by the paper.

**Output:** Results will be saved in `results/cmaes_comparison_YYYYMMDD_HHMMSS/` with COCO-compliant `.dat` files in `coco_logs/` subdirectory.

### Step 5: Download COCO Archive Data

Download baseline algorithm datasets from the [COCO Data Archive](https://coco-platform.org/testsuites/bbob/data-archive.html):

1. Visit: https://coco-platform.org/testsuites/bbob/data-archive.html
2. Download datasets for comparison algorithms:
   - **CMA-ES-2019**: Look for "CMA-ES" or "CMA-ES-2019" in the archive
   - **LQ-CMA-ES**: Look for "LQ-CMA-ES" or "CMA-ES-LQ"
   - **DTS-CMA-ES**: Look for "DTS-CMA-ES" or "CMA-ES-DTS"
   - **LMM-CMA-ES**: Look for "LMM-CMA-ES" or "CMA-ES-LMM"
3. Extract downloaded `.tgz` files
4. Place extracted folders in the `compares/` directory:
   ```
   compares/
   ├── AFN-CMA-ES/          # Your results (from Step 4)
   ├── CMA-ES-2019/          # Downloaded archive data
   ├── LQ-CMA-ES/            # Downloaded archive data
   ├── DTS-CMA-ES/           # Downloaded archive data
   └── LMM-CMA-ES/           # Downloaded archive data
   ```

**Important:** 
- Rename your AFN-CMA-ES results folder to match the algorithm name format
- Ensure folder names match exactly: `AFN-CMA-ES`, `CMA-ES-2019`, `LQ-CMA-ES`, `DTS-CMA-ES`, `LMM-CMA-ES`

### Step 6: Run COCO Post-Processing

Navigate to the `compares/` directory and run COCO post-processing:

```bash
cd compares
python -m cocopp 'AFN-CMA-ES' 'CMA-ES-2019' 'LQ-CMA-ES' 'DTS-CMA-ES' 'LMM-CMA-ES'
```

**Command Explanation:**
- `python -m cocopp`: Run COCO post-processing tool
- `'AFN-CMA-ES' 'CMA-ES-2019' ...`: List of algorithm folder names to compare
- This generates standardized comparison plots and tables

**Output:** 
- Results are saved in `compares/ppdata/` directory
- An `index.html` file is created - **open it in your browser** to view interactive comparison plots
- ECDF (Empirical Cumulative Distribution Function) plots show performance across all functions
- Comparison tables show detailed statistics

### Directory Structure

```
project/
├── .venv/                          # Virtual environment (created in Step 1)
├── compares/                       # Comparison data directory
│   ├── AFN-CMA-ES/                 # Your AFN-CMA-ES results
│   │   └── coco_logs-XXXX/         # COCO observer output files
│   │       └── data_fN/            # Function-specific data
│   │           └── *.dat          # COCO data files
│   ├── CMA-ES-2019/                # Downloaded archive data
│   ├── LQ-CMA-ES/                  # Downloaded archive data
│   ├── DTS-CMA-ES/                 # Downloaded archive data
│   ├── LMM-CMA-ES/                 # Downloaded archive data
│   └── ppdata/                     # COCO post-processing output
│       ├── index.html              # Open this in browser!
│       ├── *.pdf                   # Comparison plots
│       └── *.tex                   # LaTeX tables
├── results/                        # Raw experiment results
│   └── cmaes_comparison_YYYYMMDD_HHMMSS/
│       └── coco_logs/              # COCO-compliant output
├── run_afn.py                      # Main experiment script
└── requirements.txt                # Python dependencies
```

### Available Algorithms

- `AFN`: Standalone AFN algorithm
- `AFN-CMA-ES`: CMA-ES with AFN MLP surrogate (your proposed algorithm)
- `CMA-ES-2019`: Standard CMA-ES (from COCO archive)
- `LQ-CMA-ES`: CMA-ES with Linear-Quadratic surrogate (from COCO archive)
- `DTS-CMA-ES`: CMA-ES with Dynamic Threshold Selection (from COCO archive)
- `LMM-CMA-ES`: CMA-ES with Local Meta-Model (from COCO archive)

## 💡 Usage Examples

### Example 1: Basic AFN Optimization

```python
from afn.afn_core import AFNCore
import numpy as np

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Set up AFN
bounds = [(-5, 5), (-5, 5)]  # 2D problem
afn = AFNCore(
    input_dim=2,
    bounds=bounds,
    max_evaluations=100,
    n_models=5
)

# Optimize
result = afn.optimize(sphere, verbose=True)
print(f"Best solution: {result['best_x']}")
print(f"Best value: {result['best_y']}")
```

### Example 2: AFN-CMA-ES on BBOB Function

```python
from data.sample import load_bbob_function
from afn.cmaes_variants import AFN_CMA

# Load BBOB function
problem, info = load_bbob_function(func_id=1, dimension=2, instance=1)
print(info)

# Set up AFN-CMA-ES with BBOB bounds
bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(2)]
afn_cma = AFN_CMA(bounds=bounds, max_evaluations=100, model_type='mlp')

# Optimize
result = afn_cma.optimize(problem, verbose=True)
print(f"Best solution: {result['best_x']}")
print(f"Best value: {result['best_y']}")
```

### Example 3: BBOB Function Testing

```python
from data.sample import load_bbob_function
from afn.afn_core import AFNCore

# Load BBOB function
problem, info = load_bbob_function(func_id=1, dimension=2, instance=1)
print(info)

# Set up AFN with BBOB bounds
bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(2)]
afn = AFNCore(input_dim=2, bounds=bounds, max_evaluations=100)

# Optimize
result = afn.optimize(problem, verbose=True)
```

## 📁 Project Structure

```
final-repo/
├── afn/                                    # AFN framework core
│   ├── __init__.py                         # Package initialization
│   ├── afn_core.py                         # Main AFN implementation
│   ├── cmaes_variants.py                   # CMA-ES variants with surrogate models
│   └── simple_test_functions.py            # Standard test functions (no COCO)
├── data/                                   # Data utilities
│   ├── __init__.py                         # Package initialization
│   └── sample.py                           # BBOB data sampling functions
├── utils/                                  # Utility functions
│   ├── __init__.py                         # Package initialization
│   ├── metrics.py                          # Performance metrics calculation
│   ├── plotting.py                         # Visualization and plotting
│   └── helpers.py                          # Helper functions
├── models/                                 # Model implementations
│   └── __init__.py                         # Package initialization
├── run_afn.py                              # AFN-CMA-ES experiments (main script)
├── test_afn.py                             # Test suite
├── example_usage.py                        # Usage examples
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## 🧠 Algorithm Details

### AFN Architecture

The AFN uses a **5-model Random Forest ensemble** with the following characteristics:

- **Surrogate Models**: 5 Random Forest Regressors (100 trees each)
- **Uncertainty Estimation**: Ensemble variance across predictions
- **Input Layer**: Variable dimension (based on problem)
- **Hyperparameters**: 
  - `n_estimators=100` per Random Forest
  - `n_jobs=-1` for parallel processing
  - `random_state` for reproducibility

### Algorithm Flow

1. **Initial Sampling**: Generate 30-200 random samples (adaptive based on dimension)
2. **Surrogate Training**: Train 5-model Random Forest ensemble on collected data
3. **Candidate Selection**: 
   - Generate 1000-2000 candidate points
   - Find potential optima (lowest predictions)
   - Find high-uncertainty regions (std > 0.03)
   - Select 8 best points combining both criteria
4. **Expensive Evaluation**: Evaluate selected points
5. **Update & Repeat**: Update best solution and repeat until convergence

### Paper Specifications

- **Uncertainty threshold**: 0.03
- **Batch size**: 8 new points per iteration
- **Max evaluations**: 100
- **Convergence**: improvement < 10^-6 over 10 consecutive evaluations
- **Ensemble**: 5 Random Forest models with 100 trees each

### Comparison Algorithms

#### CMA-ES Variants

1. **Standard CMA-ES**
   - Pure CMA-ES without surrogate assistance
   - Population-based evolution strategy
   - Adaptive covariance matrix adaptation

2. **AFN-CMA-ES**
   - CMA-ES integrated with AFN ensemble surrogate
   - Supports Random Forest or MLP Deep Ensemble
   - Uncertainty-guided candidate selection

3. **LQ-CMA-ES** (Linear-Quadratic CMA-ES)
   - CMA-ES with linear-quadratic surrogate model
   - Fast global approximation
   - Reference: Hansen et al., 2019

4. **DTS-CMA-ES** (Dynamic Threshold Selection)
   - Dynamic threshold adaptation for surrogate usage
   - Balances real and surrogate evaluations
   - Reference: Bajer et al., 2019

5. **LMM-CMA-ES** (Local Meta-Model)
   - Local surrogate models around promising regions
   - Adaptive trust regions
   - Reference: Loshchilov et al., 2012

## 📊 Available Test Functions (COCO/BBOB Benchmark Suite)

| ID | Function Name | Description | Type |
|----|---------------|-------------|------|
| 1 | Sphere | Unimodal, smooth | Separable |
| 8 | Rosenbrock | Valley-shaped, deceptive | Non-separable |
| 23 | Katsuura | Highly multimodal, rugged | Non-separable |

**Note**: We use the standardized COCO/BBOB benchmark functions, which are the gold standard for optimization algorithm evaluation in academic research.

## 📈 Results & Metrics

### Performance Metrics

The comparison computes 5 key metrics:

1. **Convergence Speed**: Evaluations needed to reach 95% of best solution
2. **Optimization Accuracy**: How close to the true optimum
3. **Resource Utilization**: Efficiency of evaluation usage
4. **Exploitation Balance**: Balance between exploration and exploitation
5. **Robustness**: Consistency across multiple runs

### Generated Outputs

#### AFN-CMA-ES Experiment Output

```
results/cmaes_comparison_YYYYMMDD_HHMMSS/
├── coco_logs/                   # COCO-compliant output files
│   └── coco_logs-XXXX/          # Individual run logs
│       └── data_fN/             # Function-specific data
│           └── *.dat            # COCO data files
└── coco_archive/                 # Placeholder for COCO archive datasets
```

**After running cocopp (Step 6):**
```
compares/ppdata/
├── index.html                    # Open this in browser!
├── *.pdf                         # Comparison plots (ECDF, etc.)
└── *.tex                         # LaTeX comparison tables
```

### Command Parameters

#### AFN-CMA-ES Experiment (`run_afn.py`)

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--algorithms` | Algorithms to run | `AFN,CMA-ES` | `AFN-CMA-ES` |
| `--functions` | Test function IDs | `1,2,3` | `1-24`, `8,23` |
| `--dimensions` | Problem dimensions | `2,5` | `2,5,10,20,40` |
| `--n_runs` | Runs per test case | `10` | `15`, `30` |
| `--max_evals` | Max evaluations | `200` | `5000`, `10000` |
| `--model_type` | Surrogate model | `mlp` | `mlp` (paper requirement) |
| `--output_dir` | Results directory | `results` | `my_results` |
| `--verbose` | Detailed output | `False` | Flag |

### Expected Results

Based on the ensemble implementation, AFN-CMA-ES should demonstrate:

- **Efficient optimization** using MLP surrogate models
- **Higher optimization accuracy** across different BBOB functions
- **Better resource utilization** with fewer expensive evaluations
- **Improved robustness** across multiple runs
- **COCO-compliant results** for fair comparison with baseline algorithms

## 🐛 Troubleshooting

### Common Issues

1. **Virtual Environment Not Activated**: Make sure you're in an activated virtual environment
   ```bash
   # Check if virtual environment is active (should see (.venv) in prompt)
   # If not, activate it:
   # Windows: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```

2. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

3. **COCO Installation Issues**: Use the simple version without COCO
   ```bash
   # Test simple functions instead
   python -c "from afn.simple_test_functions import test_all_functions; test_all_functions()"
   ```

4. **Memory Issues**: Reduce batch size or max evaluations for large problems

5. **Scikit-learn Version Issues**: Ensure scikit-learn >= 1.1.0
   ```bash
   pip install --upgrade scikit-learn
   ```

6. **Long Running Times**: Use smaller `--n_runs` and `--max_evals` parameters

### Getting Help

- Check ensemble installation: `python -c "import sklearn; print(sklearn.__version__)"`
- Test individual components: `python -c "from afn import AFNCore; print('✅ Ready!')"`
- Use verbose mode: `--verbose` flag in commands
- Start with quick test: `--functions 1 --dimensions 2 --n_runs 3`

## 🔬 Advanced Usage

### Custom Surrogate Models

You can extend the EnsembleRegressor class to use different regressor models:

```python
from sklearn.ensemble import GradientBoostingRegressor
from afn.afn_core import EnsembleRegressor

class CustomEnsemble(EnsembleRegressor):
    def __init__(self, input_dim: int, n_models: int = 5, random_state: int = 42):
        super().__init__(input_dim, n_models, random_state)
        # Replace with your preferred model
        for i in range(n_models):
            model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=random_state + i
            )
            self.models.append(model)
```

### Performance Tuning

Adjust AFN parameters for better performance:

```python
afn = AFNCore(
    input_dim=dimension,
    bounds=bounds,
    uncertainty_threshold=0.05,  # Higher = more exploration
    batch_size=16,              # Larger batches for faster convergence
    max_evaluations=200,        # More evaluations for better results
    n_models=10,                # More models for better uncertainty estimation
)
```

## 📚 References

- **Paper**: "Artificial neural networks as surrogate models in optimization"
- **BBOB Suite**: [COCO/BBOB Benchmark](https://coco.gforge.inria.fr/)
- **Scikit-learn**: [Official Documentation](https://scikit-learn.org/)

## 🤝 Contributing

This implementation follows the paper specifications exactly. For modifications or improvements:

1. Maintain compatibility with existing interfaces
2. Update tests accordingly
3. Document any changes to the core algorithm
4. Ensure scikit-learn version compatibility

## 📄 License

This project is for research and educational purposes. Please cite the original paper if using this implementation in your research.

---

**🎉 Ready to run AFN optimization!**

Quick start command:
- **AFN-CMA-ES experiment**: `python run_afn.py --algorithms AFN-CMA-ES --functions 1-24 --dimensions 2,5,10,20,40 --n_runs 15 --max_evals 5000 --verbose`