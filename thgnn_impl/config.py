
# Configuration for THGNN Implementation

import torch

# Model Hyperparameters
SEQ_LEN = 30
D_MODEL = 128
N_HEADS = 8
GAT_LAYERS = 3
GAT_HEADS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-4 #0.0002 used in paper, but 1e-3/1e-4 is safer start
EPOCHS = 2
BATCH_SIZE = 1 # We process one graph (day) at a time, gradient accumulation can be used if needed
GRADIENT_ACCUMULATION_STEPS = 18

# Data Configuration
# Using a subset of S&P 500 for demonstration purposes as full S&P 500 data fetching might be slow/rate-limited
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 
    'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'PFE', 'DIS', 'KO',
    'CSCO', 'PEP', 'CMCSA', 'VZ', 'ADBE', 'CVX', 'MRK', 'INTC', 'WFC', 'T'
]

START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

# Split
TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01' 
TEST_END = '2024-12-31'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature Configuration
# List of features to be computed (37 features mentioned in paper)
# This is a placeholder list; actual implementation in features.py will define them dynamically
N_FEATURES = 37 
