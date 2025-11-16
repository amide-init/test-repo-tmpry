# Experiments Directory

This directory contains comprehensive experimental analysis files for the AFN optimization algorithm comparison study. Each file follows a structured format suitable for research paper sections.

## File Structure

### 1. Individual Metric Analyses
- **`optimization_accuracy_analysis.md`** - Detailed analysis of optimization accuracy performance
- **`convergence_speed_analysis.md`** - Comprehensive convergence speed evaluation
- **`comprehensive_metrics_analysis.md`** - Integrated analysis of all five performance metrics
- **`afn_hansen_bajer_comparison_summary.md`** - Comprehensive comparison with state-of-the-art Hansen and Bajer algorithms

### 2. Analysis Format
Each analysis file follows this research paper structure:

1. **What is [Metric]?** - Definition and importance
2. **Formula and Mathematical Explanation** - Mathematical formulation with detailed explanation
3. **Experimental Design** - Methodology, parameters, and setup
4. **Experimental Results Visualization** - Graphs and figures with captions
5. **Results and Analysis** - Detailed findings, statistical analysis, and implications

### 3. Key Features

#### Evaluation Budget Analysis
All experiments include analysis across multiple evaluation budgets:
- **10 evaluations**: Ultra-limited budget
- **40 evaluations**: Low budget  
- **80 evaluations**: Medium budget
- **120 evaluations**: High budget
- **200 evaluations**: Full budget

#### Algorithm Comparison
Comprehensive comparison of optimization algorithms:

**Primary Comparison (AFN vs Traditional Methods):**
- **AFN (Adaptive Fidelity Nexus)**: Proposed surrogate-based method
- **GA (Genetic Algorithm)**: Population-based evolutionary approach
- **PSO (Particle Swarm Optimization)**: Swarm intelligence method
- **ACO (Ant Colony Optimization)**: Metaheuristic optimization

**State-of-the-Art Comparison (AFN vs Advanced Methods):**
- **AFN (Adaptive Function Network)**: Proposed method
- **Hansen (CMA-ES)**: Covariance Matrix Adaptation Evolution Strategy
- **Bajer (Gaussian Process)**: GP-based optimization with Expected Improvement

#### Benchmark Functions
Standardized evaluation using COCO/BBOB benchmark suite:
- **Function 1 (Sphere)**: Unimodal, separable function
- **Function 8 (Rosenbrock)**: Non-separable, valley-shaped function
- **Function 23 (Katsuura)**: Highly multimodal, non-separable function

#### Performance Metrics
Five critical metrics for comprehensive evaluation:
1. **Optimization Accuracy**: Solution quality relative to global optimum
2. **Convergence Speed**: Computational efficiency in reaching target performance
3. **Resource Utilization**: Effective use of evaluation budget
4. **Exploitation Balance**: Exploration vs exploitation strategy effectiveness
5. **Robustness**: Consistency and reliability across multiple runs

## Usage for Research Paper

These files are designed to be directly integrated into research paper sections:

- **Section 3**: Use individual metric analyses for detailed methodology
- **Section 4**: Use visualization references for experimental results
- **Section 5**: Use results and analysis for discussion and conclusions
- **Appendix**: Use comprehensive metrics for detailed statistical analysis

## Future Extensions

Additional analysis files can be added following the same format:
- **`exploitation_balance_analysis.md`** - Detailed exploitation/exploration analysis
- **`robustness_analysis.md`** - Comprehensive robustness evaluation
- **`resource_utilization_analysis.md`** - Budget efficiency analysis
- **`afn_hansen_bajer_comparison_summary.md`** - ✅ **COMPLETED** - Comparison with state-of-the-art methods

## Data Sources

All experimental data is generated using:
- **Primary Comparison Script**: `run_afn_ga_pso_aco_comparison.py`
- **State-of-the-Art Comparison Script**: `run_afn_hansen_bajer_comparison.py`
- **Results Directory**: `results/` containing JSON metrics and visualization files
- **BBOB Functions**: COCO/BBOB benchmark suite for standardized evaluation

## Citation Format

When using these analyses in your research paper, please cite:
- The specific metric analysis file
- The experimental methodology
- The BBOB benchmark functions
- The statistical analysis methods used
