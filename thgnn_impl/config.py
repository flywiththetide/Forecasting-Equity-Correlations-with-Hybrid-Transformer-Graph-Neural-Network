
# Configuration for THGNN Implementation

import torch

# Model Hyperparameters
SEQ_LEN = 30
D_MODEL = 128
N_HEADS = 8
GAT_LAYERS = 3
GAT_HEADS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-4 
EPOCHS = 2
BATCH_SIZE = 1 
GRADIENT_ACCUMULATION_STEPS = 18

# Data Configuration
# Expanded to ~300 tickers (approx 10x of previous 30)
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 
    'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'PFE', 'DIS', 'KO',
    'CSCO', 'PEP', 'CMCSA', 'VZ', 'ADBE', 'CVX', 'MRK', 'INTC', 'WFC', 'T',
    'ABT', 'CRM', 'ABBV', 'NKE', 'MCD', 'TMO', 'DHR', 'ACN', 'NFLX', 'LIN',
    'TXN', 'COST', 'PM', 'NEE', 'BMY', 'ORCL', 'HON', 'AMGN', 'UNP', 'LOW',
    'SPGI', 'UPS', 'IBM', 'QCOM', 'CAT', 'MS', 'GE', 'DE', 'GS', 'LMT',
    'MMM', 'INTU', 'AXP', 'CVS', 'BLK', 'SBUX', 'AMT', 'NOW', 'AMD', 'ISRG',
    'SCHW', 'BKNG', 'ADP', 'MDLZ', 'GILD', 'PLD', 'ZTS', 'TGT', 'SYK', 'TJX',
    'C', 'MMC', 'CB', 'LRCX', 'MO', 'TMUS', 'PGR', 'SO', 'SLB', 'DUK'
]

START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

TRAIN_START = '2019-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01' 
TEST_END = '2024-12-31'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_FEATURES = 37 
