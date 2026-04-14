"""
Synthetic Credit Card Fraud Dataset Generator
==============================================
Generates a dataset structurally identical to the Kaggle Credit Card Fraud
Detection dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Schema:
  Time   – seconds elapsed since first transaction in the window
  V1-V28 – PCA-transformed features (anonymized for privacy)
  Amount – transaction amount in USD
  Class  – 0 = legitimate, 1 = fraudulent
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset size defaults (approx. Kaggle dataset proportions)
DEFAULT_N_LEGIT = 50000
DEFAULT_N_FRAUD = 500   # ~1% fraud rate for demo (real dataset is 0.17%)


def generate_creditcard_data(
    n_legit: int = DEFAULT_N_LEGIT,
    n_fraud: int = DEFAULT_N_FRAUD,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data with realistic fraud patterns.

    The fraud transactions embed statistically anomalous signatures in specific
    PCA components (V1, V3, V4, V10, V12, V14, V17) that mirror patterns
    observed in real-world card fraud data.

    Args:
        n_legit:       Number of legitimate transactions to generate.
        n_fraud:       Number of fraudulent transactions to generate.
        random_state:  Seed for reproducibility.

    Returns:
        pd.DataFrame with columns [Time, V1-V28, Amount, Class], shuffled.
    """
    np.random.seed(random_state)
    logger.info(f"Generating {n_legit:,} legitimate + {n_fraud:,} fraud transactions...")

    # ── Legitimate transactions ───────────────────────────────────────────────
    legit_V = np.random.multivariate_normal(
        mean=np.zeros(28),
        cov=np.eye(28) * 2.0,
        size=n_legit,
    )
    # Realistic amount distribution: mostly small purchases, heavy right tail
    legit_amount = np.abs(np.random.exponential(scale=60.0, size=n_legit))
    legit_amount = np.clip(legit_amount, 0.01, 10000.0)
    legit_time = np.sort(np.random.uniform(0, 172_800, n_legit))  # 48-hour window

    # ── Fraudulent transactions ───────────────────────────────────────────────
    fraud_V = np.random.multivariate_normal(
        mean=np.zeros(28),
        cov=np.eye(28) * 1.5,
        size=n_fraud,
    )
    # Embed distinct fraud signatures in key PCA components
    fraud_V[:, 0]  = np.random.uniform(-10, -3,  n_fraud)   # V1:  strong negative
    fraud_V[:, 2]  = np.random.uniform(-10, -1,  n_fraud)   # V3:  strong negative
    fraud_V[:, 3]  = np.random.uniform(1,   8,   n_fraud)   # V4:  positive spike
    fraud_V[:, 9]  = np.random.uniform(-6,  -1,  n_fraud)   # V10: negative
    fraud_V[:, 11] = np.random.uniform(-8,  -3,  n_fraud)   # V12: negative
    fraud_V[:, 13] = np.random.uniform(-12, -4,  n_fraud)   # V14: strongest signal
    fraud_V[:, 16] = np.random.uniform(-8,  -2,  n_fraud)   # V17: negative

    # Fraud amounts: bimodal — small (card testing) and large (cash-out)
    fraud_amount_large = np.abs(np.random.normal(loc=300.0, scale=350.0, size=n_fraud // 2))
    fraud_amount_small = np.random.uniform(0.01, 5.0, size=n_fraud - n_fraud // 2)
    fraud_amount = np.concatenate([fraud_amount_large, fraud_amount_small])
    np.random.shuffle(fraud_amount)
    fraud_amount = np.clip(fraud_amount, 0.01, 25_000.0)
    fraud_time = np.random.uniform(0, 172_800, n_fraud)

    # ── Combine ───────────────────────────────────────────────────────────────
    V_all   = np.vstack([legit_V, fraud_V])
    amounts = np.concatenate([legit_amount, fraud_amount])
    times   = np.concatenate([legit_time, fraud_time])
    labels  = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(V_all, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"]  = labels.astype(int)

    # Shuffle to prevent ordering bias
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    fraud_rate = df["Class"].mean() * 100
    logger.info(f"Dataset ready: {len(df):,} rows | Fraud rate: {fraud_rate:.2f}%")
    logger.info(f"Amount stats: mean=${df['Amount'].mean():.2f} | max=${df['Amount'].max():.2f}")

    return df


if __name__ == "__main__":
    df = generate_creditcard_data()
    output_path = os.path.join(os.path.dirname(__file__), "creditcard.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"\nClass distribution:\n{df['Class'].value_counts()}")
    print(f"\nSample:\n{df.head(3).to_string()}")
