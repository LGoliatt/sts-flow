# STS-FLOW

**Stacked Two-Stage Flowing Bottomhole Pressure Learning and Oilfield Wellbore Predictor**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

STS-FLOW is an advanced machine learning framework for predicting Flowing Bottomhole Pressure (FBHP) and other critical oilfield parameters using a sophisticated two-stage stacking ensemble approach. The system combines multiple regression algorithms through hierarchical meta-learning to achieve superior prediction accuracy for petroleum reservoir engineering applications.

### Key Features

- **Two-Stage Stacking Architecture**: Hierarchical ensemble combining non-linear (SVR, KNN, ELM) and traditional models (Linear Regression, Random Forest)
- **Multiple Base Learners**: Includes SVR, KNN, Random Forest, ELM, and more
- **Comprehensive Evaluation**: RMSE, R², Pearson correlation, and KGE metrics
- **Uncertainty Quantification**: Monte Carlo-based uncertainty estimation
- **Feature Importance Analysis**: Permutation-based feature importance assessment
- **Production Ready**: Robust error handling and reproducible results with seed control

## Architecture

```
STS-FLOW Model Hierarchy:
├── ST-S (Two-Stage Stacking)
│   ├── ML-1 (Non-linear Meta-Learner)
│   │   ├── SVR (Support Vector Regression)
│   │   ├── KNN (K-Nearest Neighbors)
│   │   └── ELM (Extreme Learning Machine)
│   ├── ML-2 (Traditional Meta-Learner)
│   │   ├── Linear Regression
│   │   └── Random Forest
│   └── Final Estimator: RidgeCV
├── ST-1, ST-2, ST-3 (Single-Stage Stacking variants)
└── Base Models (Individual regressors)
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Dependencies

```bash
pip install numpy pandas scikit-learn scipy statsmodels
pip install xgboost py-earth hydroeval
pip install gmdhpy
```

### Data Reader Modules

Ensure the following data reader modules are available:
- `shammari_read_data.py`

## Usage

### Basic Usage

Run the main script with default parameters (100 runs):

```bash
python fbhp_stacking.py
```

### Dataset Selection

Modify the `datasets` list in the script to select your target dataset:

```python
datasets = [
    read_shammari(target='FBHP', dataset='FBHP-2', seed=run, plot=True),
    # Add other datasets as needed
]
```

## Model Configurations

### ST-S (Primary Two-Stage Model)

The ST-S model uses:
- **First Stage**: Two parallel meta-learners (ML-1 and ML-2)
- **Second Stage**: RidgeCV with 500 alpha values
- **Cross-Validation**: 5-fold CV
- **Parallel Processing**: Multi-core support via n_jobs=-1

### Available Models

| Model | Description |
|-------|-------------|
| SVR | Support Vector Regression |
| LR | Linear Regression |
| KNN | K-Nearest Neighbors |
| RF | Random Forest (30 estimators) |
| ELM | Extreme Learning Machine |
| ST-1 | Stacking with Linear Regression |
| ST-2 | Stacking with RidgeCV |
| ST-3 | Stacking with LassoCV |
| ST-S | Two-Stage Hierarchical Stacking |

## Output

### Results Structure

Results are saved as JSON files in the `pkl_stck_elm_<dataset>/` directory:

```
pkl_stck_elm_fbhp-2/
└── stck_elm_run_0.json
└── stck_elm_run_1.json
└── ...
```

### Metrics Provided

- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of Determination
- **R**: Pearson Correlation Coefficient
- **KGE**: Kling-Gupta Efficiency
- **MPE**: Mean Prediction Error
- **WUB**: Within Uncertainty Bounds
- **PEI95**: 95% Prediction Error Interval
- **Feature Importance**: Permutation-based rankings

### Uncertainty Quantification

The framework generates 125,000 Monte Carlo samples to estimate:
- Median prediction
- Mean Absolute Deviation (MAD)
- Relative uncertainty percentage

## Visualization

The script automatically generates:
- Scatter plots of observed vs. predicted values
- Performance metrics overlays
- Feature importance plots (when enabled)

## Example Output

```
>> 0 fbhp-2    ST-S            1.234       0.956       0.978
```

Where:
- `0`: Run number
- `fbhp-2`: Dataset name
- `ST-S`: Model name
- `1.234`: RMSE
- `0.956`: R²
- `0.978`: Pearson R

## Configuration

### Random Seed

Each run uses a deterministic seed: `random_seed = run + 37`

### Feature Scaling

All features are normalized using MinMaxScaler (0-1 range) before model training.

### Cross-Validation

5-fold cross-validation is applied in all stacking regressors.

## Advanced Features

### Feature Importance Analysis

```python
from sklearn.inspection import permutation_importance
permut_imp = permutation_importance(est, X_test_scaled, y_test, 
                                    n_repeats=30, random_state=0)
```

### Custom Alpha Grid

Modify the alpha values for Ridge/Lasso regularization:

```python
alphas = np.logspace(-6, 6, 25)  # Logarithmic spacing
alphas = np.linspace(1e-6, 10, 500)  # Linear spacing
```

## Performance Optimization

- **Parallel Processing**: Set `n_jobs=-1` for maximum CPU utilization
- **Memory Management**: Results saved incrementally after each model
- **Warning Suppression**: Non-critical warnings filtered for cleaner output

## Troubleshooting

### Common Issues

1. **Module Not Found**: Ensure all custom utility modules are in the `util/` directory
2. **Memory Error**: Reduce `n_outcomes` in uncertainty quantification
3. **Convergence Warnings**: Increase `max_iter` in iterative models

### Debug Mode

Remove warning suppression to see detailed output:

```python
# Comment out these lines:
# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")
```

## Citation

If you use STS-FLOW in your research, please cite:

```bibtex
@software{sts_flow,
  title={STS-FLOW: Stacked Two-Stage Flowing Bottomhole Pressure Learning and Oilfield Wellbore Predictor},
  author={Leonardo Goliatt},
  year={2025},
  url={https://github.com/yourusername/sts-flow}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed for petroleum reservoir engineering applications
- Built on scikit-learn's ensemble learning framework
- Incorporates custom implementations of ELM, RBFNN, and LSSVR

## Contact

For questions, issues, or collaboration opportunities:
- GitHub Issues: [Create an issue](https://github.com/yourusername/sts-flow/issues)
- Email: leonardo.goliatt@ufjf.br

## References

- Shammari et al. - FBHP Dataset
- Nwanwe et al. - FBHP Modeling
- Scikit-learn Documentation: [Stacking Ensemble](https://scikit-learn.org/stable/modules/ensemble.html#stacking)

---
