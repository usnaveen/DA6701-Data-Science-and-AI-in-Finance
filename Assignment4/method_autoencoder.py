"""method_autoencoder.py — Autoencoder-based portfolio replication (Method 2).

Strategy
--------
1. Train a symmetric autoencoder on the T × N daily return matrix.
2. Compute a *communality* score for each stock: the R² of reconstructing
   that stock's return series from the bottleneck latent representation.
3. Rank stocks by communality and select the top-k.
4. Solve a Quadratic Programme (QP) to find long-only weights that minimise
   the variance of active returns (portfolio − benchmark).
5. Sweep k ∈ {10, 15, …, 100} and report val + holdout metrics.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from eval import evaluate, load_splits


class ReturnAutoencoder(nn.Module):
    """Symmetric autoencoder: N → 128 → latent_dim → 128 → N."""

    def __init__(self, n_stocks: int, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_stocks, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_stocks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    R_train: pd.DataFrame,
    latent_dim: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    val_frac: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
) -> ReturnAutoencoder:
    """Train the autoencoder with early stopping on a held-out val split.

    Returns the trained model (moved to CPU).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_np = R_train.fillna(0.0).values.astype(np.float32)
    n_total = len(X_np)
    n_val = max(1, int(n_total * val_frac))
    X_tr = torch.tensor(X_np[:-n_val], device=device)
    X_vl = torch.tensor(X_np[-n_val:], device=device)

    model = ReturnAutoencoder(n_stocks=X_np.shape[1], latent_dim=latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        loss = criterion(model(X_tr), X_tr)
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), X_vl).item()

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best val loss={best_val_loss:.6f})")
                break

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={loss.item():.6f} | val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


def compute_communality(
    model: ReturnAutoencoder, R_train: pd.DataFrame
) -> dict:
    """Return per-stock R² (communality) from the trained autoencoder."""
    X_np = R_train.fillna(0.0).values.astype(np.float32)
    model.eval()
    with torch.no_grad():
        R_reconstructed = model(torch.tensor(X_np)).numpy()

    communality = {}
    for i, ticker in enumerate(R_train.columns):
        communality[ticker] = r2_score(X_np[:, i], R_reconstructed[:, i])
    return communality


def fit_weights_qp(
    selected_tickers: list,
    R_train: pd.DataFrame,
    b_train: pd.Series,
) -> dict:
    """Solve a QP to minimise variance of active returns (long-only, sum=1).

    Falls back to clipped OLS if the optimiser fails to converge.
    """
    X = R_train[selected_tickers].fillna(0.0).values
    y = b_train.values
    n = len(selected_tickers)

    def objective(w):
        active = X @ w - y
        return active.var()

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})

    if res.success:
        w_final = res.x
    else:
        from numpy.linalg import lstsq
        w_ols, _, _, _ = lstsq(X, y, rcond=None)
        w_ols = np.clip(w_ols, 0, None)
        total = w_ols.sum()
        w_final = w_ols / total if total > 0 else np.ones(n) / n

    return dict(zip(selected_tickers, w_final.tolist()))


def run_autoencoder_sweep(
    R_train: pd.DataFrame,
    b_train: pd.Series,
    R_val: pd.DataFrame,
    b_val: pd.Series,
    R_hold: pd.DataFrame,
    b_hold: pd.Series,
    latent_dim: int = 32,
    n_epochs: int = 100,
    k_range: range | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, ReturnAutoencoder]:
    """Train autoencoder, rank by communality, sweep k, return results.

    Returns
    -------
    ae_results : pd.DataFrame
        One row per k with val/holdout metrics.
    communality : dict
        Per-stock R² scores.
    model : ReturnAutoencoder
        The trained model (useful for saving/inspection).
    """
    if k_range is None:
        k_range = range(10, 101, 5)

    if verbose:
        print("Training autoencoder …")
    model = train_autoencoder(
        R_train, latent_dim=latent_dim, n_epochs=n_epochs, verbose=verbose
    )

    if verbose:
        print("Computing communality scores …")
    communality = compute_communality(model, R_train)
    ranked = sorted(communality.items(), key=lambda x: -x[1])

    records = []
    for k in k_range:
        top_k = [t for t, _ in ranked[:k]]
        weights = fit_weights_qp(top_k, R_train, b_train)
        res = evaluate(weights, R_val, b_val, R_hold, b_hold, label="Autoencoder")
        res["weights"] = weights
        records.append(res)
        if verbose:
            print(f"  k={k:3d} | TE_val={res['TE_val']:.4f} | TE_hold={res['TE_hold']:.4f}")

    ae_results = pd.DataFrame(records)
    return ae_results, communality, model


def get_ae_weights_for_k(results_df: pd.DataFrame, k: int = 50) -> dict:
    """Extract weight dict for the closest k in results_df."""
    if results_df.empty:
        raise ValueError("results_df is empty — run run_autoencoder_sweep first.")
    row = results_df.iloc[(results_df["k"] - k).abs().argsort()[:1]]
    return row["weights"].iloc[0]


if __name__ == "__main__":
    R_train, b_train, R_val, b_val, R_hold, b_hold = load_splits()
    print(f"Running Autoencoder sweep on {R_train.shape[1]} stocks …")
    results, communality, model = run_autoencoder_sweep(
        R_train, b_train, R_val, b_val, R_hold, b_hold
    )
    print(results[["k", "TE_val", "IR_val", "TE_hold", "IR_hold"]].to_string())
    results.drop(columns="weights").to_csv("data/ae_results.csv", index=False)
    print("Saved → data/ae_results.csv")
