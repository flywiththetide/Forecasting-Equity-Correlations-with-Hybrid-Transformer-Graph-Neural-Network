
# THGNN: Transformer-Heterogeneous Graph Neural Network for Equity Correlation Forecasting

This project implements the THGNN architecture described in "Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network". It forecasts 10-day ahead stock-stock correlations using a combination of Temporal Encoders (Transformer) and Relational Encoders (GAT).

## Project Structure
- `main.py`: Entry point for training and analysis.
- `thgnn_impl/`
  - `data/`: Data fetching, feature engineering, graph construction.
  - `models/`: Transformer, GAT, Experts, Layers.
  - `training/`: Loss functions, training loop.
  - `analysis/`: Plotting, metrics, interpretation.
  - `utils/`: Logging, helpers.

## Results & Evidence

The model was trained on a subset of S&P 500 stocks. Below are the visual results of the training run.

### 1. Training Performance
The loss curve shows the convergence of the Hybrid Loss (Huber + Histogram) over epochs.
![Loss Curve](results/loss_curve.png)

### 2. Prediction Accuracy
Scatter plot of Predicted vs Actual Fisher-z correlations. Ideally, points lie on the red y=x line.
![Prediction Scatter](results/scatter_pred.png)

### 3. Correlation Matrix Comparison
Heatmap of the actual vs predicted correlation matrix for a sample day.
![Heatmap](results/heatmap_comparison.png)

### 4. Residual Distribution
Histogram of prediction errors (residuals). A centered gaussian-like distribution indicates unbiased predictions.
![Residuals](results/residual_distribution.png)

### 5. Feature Importance
Gradient x Input analysis showing the most influential features driving the predictions.
![Feature Importance](results/feature_importance.png)

### 6. Expert Usage
Frequency of usage for the Negative, Neutral, and Positive expert heads.
![Expert Usage](results/expert_usage.png)

### 7. Rolling MSE
Mean Squared Error over the validation period timeline.
![Rolling MSE](results/rolling_mse.png)

### 8. Time Series Comparison
Average market correlation (Actual vs Predicted) over time.
![Time Series](results/time_series_pair.png)

### 9. Volatility vs Error
Relationship between stock volatility and model prediction error.
![Volatility vs Error](results/error_vs_volatility.png)

### 10. Graph Degree Distribution
Distribution of node degrees in the correlation graph.
![Degree Dist](results/degree_dist.png)

## Metrics
- **MAE**: 0.405
- **RMSE**: 0.513
- **Correlation**: 0.492
