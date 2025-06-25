# Stock Market Prediction with XGBoost and PPO

This project presents a two-stage learning framework for stock market prediction and algorithmic trading using a **single Jupyter Notebook**. It integrates:

- **XGBoost** for supervised directional prediction  
- **Proximal Policy Optimization (PPO)** for reinforcement learning-based trade execution


## Notebook

The entire workflow resides in:
notebook/stock_market_prediction.ipynb

---

## Project Highlights

- **Feature Engineering**: 
  - Technical indicators (RSI, EMA, MACD, etc.)
  - Lag features, rolling stats, and collinearity pruning
- **XGBoost Classifier**:
  - Trained to predict if the next day’s price will rise
  - Outputs probabilities (`xgb_pred_prob`) fed into the PPO agent
- **PPO Agent**:
  - Learns trading behavior via a 3-action discrete space (Buy, Hold, Sell)
  - Enhanced with short-selling, episodic logic, reward shaping and asset ID awareness
- **Evaluation**:
  - Sharpe ratio, equity curve, directional alignment with benchmark signals

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

Make sure you have Python 3.9+ installed. Then:

```bash
pip install -r requirements.txt

pip install jupyterlab 
```

### 2. Launch notebook

- jupyter lab

### 3. Execute

- Open the notebook in **Jupyter Lab** and run all cells sequentially.

The following output files will be generated:

- `multi_ticker_data.csv`: Contains the extracted data, from which features will be engineered.  
- `features_with_xgb_pred.csv`: Stores the features along with the `xgb_pred_prob` output from XGBoost, which will be fed into PPO.  
- `episode_pnl_log.csv`: Logs the profit and loss (PnL) for each episode during PPO evaluation.

## Expected Output

### The notebook outputs:
	•	XGBoost metrics: ROC AUC, confusion matrix
	•	PPO metrics: PnL curve, Sharpe ratio, win/loss rate
	•	Visuals: feature importances, trading behavior, and equity progression


### License

MIT License — free to use, modify, and share with attribution.

### Author

Charles Santhakumar Sagayanathan
M.Sc. Data Science – June 2025