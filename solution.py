import argparse

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import lognorm

NIGHT_HOURS = (0, 1, 2, 3, 4, 5)
RISKY_MCC = (5964, 5967, 7995, 6051)
ROUND_CENTS = (0, 50, 99)

G1_RANGE = (0.80, 0.95)
G2_RANGE = (500.0, 5000.0)
SMALL_LIMIT = 500.0

G1_BACKGROUND = 36.0
G2_CENT_SHARE = 0.03


def read_frame(path):
    df = pd.read_csv(path)
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["cents"] = (df["amount"] * 100).round().astype("int64") % 100
    df["ratio"] = df["amount"] / df["account_balance_before"].replace(0, np.nan)
    return df


def injection_masks(df):
    g1 = (df["ratio"] >= G1_RANGE[0]) & (df["ratio"] <= G1_RANGE[1])
    g2 = (
        (~g1)
        & df["cents"].isin(ROUND_CENTS)
        & (df["amount"] >= G2_RANGE[0])
        & (df["amount"] < G2_RANGE[1])
    )
    return g1.to_numpy(), g2.to_numpy()


def fit_amount_law(df, candidate):
    amounts = df.loc[~candidate, "amount"]
    body = amounts[(amounts > 0) & (amounts < 5000)].to_numpy()
    shape, _, scale = lognorm.fit(body, floc=0)
    return shape, scale


def flag_matrix(df):
    return {
        "foreign": df["is_foreign"].to_numpy(),
        "night": df["hour"].isin(NIGHT_HOURS).to_numpy(),
        "risky": df["merchant_category"].isin(RISKY_MCC).to_numpy(),
        "unknown": (df["device_type"] == "unknown").to_numpy(),
        "desktop": (df["device_type"] == "desktop").to_numpy(),
        "mobile": (df["device_type"] == "mobile").to_numpy(),
        "tablet": (df["device_type"] == "tablet").to_numpy(),
    }


def estimate_flag_rates(df, flags, candidate, n_fraud):
    normal = ~candidate
    background = {name: float(value[normal].mean()) for name, value in flags.items()}
    seen = {name: int(value[candidate].sum()) for name, value in flags.items()}

    n_rows = int(candidate.sum())
    n_background = n_rows - n_fraud
    rate = {}
    for name in flags:
        p = (seen[name] - n_background * background[name]) / n_fraud
        rate[name] = float(np.clip(p, 0.01, 0.99))
    return rate, background


def log_likelihood_ratio(flag, p_fraud, p_normal):
    flag = np.asarray(flag, dtype=bool)
    return np.where(
        flag,
        np.log(p_fraud / p_normal),
        np.log((1.0 - p_fraud) / (1.0 - p_normal)),
    )


def build_prediction(df):
    g1, g2 = injection_masks(df)
    candidate = g1 | g2
    n = len(df)

    band2_rows = int(((df["amount"] >= G2_RANGE[0]) & (df["amount"] < G2_RANGE[1])).sum())
    bg2 = G2_CENT_SHARE * band2_rows

    n_g1 = max(g1.sum() - G1_BACKGROUND, 1)
    n_g2 = max(g2.sum() - bg2, 1)
    n_fraud = n_g1 + n_g2
    prior_g1, prior_g2 = n_g1 / n_fraud, n_g2 / n_fraud

    shape, scale = fit_amount_law(df, candidate)
    flags = flag_matrix(df)
    rate, background = estimate_flag_rates(df, flags, candidate, n_fraud)

    feature_llr = (
        log_likelihood_ratio(flags["foreign"], rate["foreign"], background["foreign"])
        + log_likelihood_ratio(flags["night"], rate["night"], background["night"])
        + log_likelihood_ratio(flags["risky"], rate["risky"], background["risky"])
        + log_likelihood_ratio(flags["unknown"], rate["unknown"], background["unknown"])
        + log_likelihood_ratio(flags["desktop"], rate["desktop"], background["desktop"])
        + log_likelihood_ratio(flags["mobile"], rate["mobile"], background["mobile"])
        + log_likelihood_ratio(flags["tablet"], rate["tablet"], background["tablet"])
    )

    amount = df["amount"].to_numpy()
    balance = df["account_balance_before"].to_numpy()

    log_normal = np.full(n, -1000.0)
    positive = amount > 0
    log_normal[positive] = lognorm.logpdf(amount[positive], shape, scale=scale)

    amt_llr_g1 = np.full(n, -np.inf)
    amt_llr_g1[g1] = -np.log(0.15 * np.abs(balance[g1])) - log_normal[g1]

    amt_llr_g2 = np.full(n, -np.inf)
    width = (G2_RANGE[1] - G2_RANGE[0]) * len(ROUND_CENTS)
    amt_llr_g2[g2] = -np.log(width) - (log_normal[g2] + np.log(0.01))

    s1 = np.full(n, -np.inf)
    s2 = np.full(n, -np.inf)
    s1[g1] = np.log(prior_g1) + amt_llr_g1[g1] + feature_llr[g1]
    s2[g2] = np.log(prior_g2) + amt_llr_g2[g2] + feature_llr[g2]

    score = np.full(n, -np.inf)
    score[candidate] = logsumexp(np.vstack([s1[candidate], s2[candidate]]), axis=0)

    pred = np.empty(n)
    cand_idx = np.flatnonzero(candidate)

    small_round = (
        (~candidate)
        & df["cents"].isin(ROUND_CENTS).to_numpy()
        & (amount < SMALL_LIMIT)
    )
    small_idx = np.flatnonzero(small_round)
    rest_idx = np.flatnonzero(~candidate & ~small_round)

    order = cand_idx[np.argsort(score[cand_idx])]
    pred[order] = 0.520 + 0.479 * np.linspace(0, 1, len(order))

    if len(small_idx):
        small_score = 0.5 * feature_llr[small_idx]
        pred[small_idx[np.argsort(small_score)]] = 0.480 + 0.035 * np.linspace(
            0, 1, len(small_idx)
        )

    rest_score = (
        0.5 * feature_llr[rest_idx]
        + 0.5 * np.clip((np.log10(np.maximum(amount[rest_idx], 1)) - 3.0) / 2.0, 0, 1)
    )
    pred[rest_idx[np.argsort(rest_score)]] = 0.001 + 0.449 * np.linspace(
        0, 1, len(rest_idx)
    )

    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="transactions.csv")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    df = read_frame(args.input)
    pred = build_prediction(df)

    pd.DataFrame(
        {"transaction_id": df["transaction_id"].to_numpy(), "prediction": pred}
    ).to_csv(args.output, index=False)
    print("wrote %s (%d rows)" % (args.output, len(df)))


if __name__ == "__main__":
    main()
