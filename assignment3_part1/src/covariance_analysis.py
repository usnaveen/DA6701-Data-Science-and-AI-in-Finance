import numpy as np
from config import DATA_PROCESSED

def compute_sample_covariance(train_returns):
    S = train_returns.cov()
    S.to_csv(DATA_PROCESSED / "sample_covariance.csv")
    return S

def compute_condition_number(S):
    return np.linalg.cond(S)