# Credit Card Fraud Detection

## Business problem
Identify fraudulent credit card transactions in real time, while understanding the financial trade-off between the two types of error: letting fraud through (false negative) vs. blocking/flagging a legitimate transaction (false positive).

## Dataset
- **Source**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle (ULB Machine Learning Group / Worldline)
- **Size**: 284,807 real European card transactions (September 2013); features `V1`-`V28` are PCA-anonymized
- **Notes**: Only 492 transactions (0.17%) are fraud — an extremely imbalanced dataset. Unlike the other two projects, individual features carry no business meaning, so the focus here is exploratory patterns + a cost-based decision analysis rather than interpretable hypotheses.

## Approach
1. Exploratory patterns in `Time` and `Amount`
2. Baseline fraud classifier (logistic regression, balanced class weights)
3. Cost-based analysis of the false positive / false negative trade-off across classification thresholds

Full detail and code in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

## Tools used
`Python` `pandas` `scikit-learn` `matplotlib`

## Key findings
- **Time-of-day pattern**: fraud rate peaks at 2am-5am (up to 1.7% of transactions) — roughly 10x the ~0.15% daytime baseline.
- **Amount is not a reliable standalone signal**: median fraud amount ($9.25) is lower than median legitimate ($22), but mean fraud amount is higher ($122 vs $88) — fraud includes many small test charges plus some large transactions.
- **Default threshold (0.5) is far from cost-optimal**: it catches 87.8% of fraud but generates 1,806 false alarms (6.7% precision) on the test set.
- **Recommended threshold (~0.98)**: still catches 83.1% of fraud, cuts false alarms by ~95% (to 96), and reduces total estimated cost by ~67% (from ~$22,721 to ~$7,488 on the test set), under a $10-per-false-positive cost assumption.

## Business recommendation
1. Move the alert threshold from the model's default (0.5) to ~0.98 — cuts false alarms by ~95% and total estimated cost by ~67%, while still catching over 80% of fraud.
2. Prioritize alert review capacity for the 2am-5am window, where fraud rate is ~10x the daily average.
3. Revisit the $10 false-positive cost assumption with real operational data — the optimal threshold is directly sensitive to it.

## Limitations
- Threshold was tuned and evaluated on the same test set; a production system should validate on a separate held-out set.
- The $10 false-positive cost is an assumption, not a measured business figure.
- Logistic regression is a baseline — the cost-optimization approach applies to any model's probability scores.
- PCA-anonymized features mean the analysis can't explain *why* a transaction looks fraudulent, only that it does.

## How to run this project
```bash
git clone https://github.com/pedronotarnicola/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

## Folder structure
```
fraud-detection/
├── data/
│   └── raw/              # creditcard.csv
├── notebooks/
│   └── analysis.ipynb
├── images/
├── README.md
└── requirements.txt
```

---
[Portfolio](https://pedronotarnicola.github.io/) · [LinkedIn](https://www.linkedin.com/in/p-l-notarnicola/)
