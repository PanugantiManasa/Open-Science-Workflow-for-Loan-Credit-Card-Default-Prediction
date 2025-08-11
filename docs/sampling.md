# Sampling Plan

## Project: Predicting Loan Default from Open Data (UCI Credit Cards + Kaggle P2P Loans)

## purpose: Make our train/test selection reproducible and auditable across both datasets.

## Research question & hypothesis

RQ: How accurately can borrower-level financial and demographic attributes predict default across different lending contexts? H1 (predictive): Stronger signs of repayment stress (e.g., recent delinquency, high payment burden) are associated with a higher probability of default.

## Datasets (sampling frames)

### UCI – Default of Credit Card Clients

Source: UCI ML Repository (Taiwan credit card clients)

Raw file: data/raw/default_of_credit_card_clients.csv

Rows: ~30,000 (binary target)

Features: 25

Feature Names: ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, default payment next month

Target (renamed in processing): target_default (from default.payment.next.month)

### Kaggle – P2P Lending (train_public)

Source: Kaggle competition dataset (Chinese lending platform)

Raw file: data/raw/train_public.csv

Rows: ~10,000 (binary target)

Features: 39

Feature Names: loan_id, user_id, total_loan, year_of_loan, interest, monthly_payment, class, employer_type, industry, work_year, house_exist, censor_status, issue_date, use, post_code, region, debt_loan_ratio, del_in_18month, scoring_low, scoring_high, known_outstanding_loan, known_dero, pub_dero_bankrup, recircle_b, recircle_u, initial_list_status, app_type, earlies_credit_mon, title, policy_code, f0, f1, f2, f3, f4, early_return, early_return_amount, early_return_amount_3mon, isDefault

Target (as provided): isDefault → renamed to target_default during processing.

Both sources are observational and anonymised; we do not add or infer personal identifiers.

## Sampling design

The design involved observational use of the complete dataset in each case, without global down-sampling. Data was split into an 80/20 stratified train–test set to preserve the default rate, with a fixed random seed of 42 for reproducibility. Model tuning employed 3-fold cross-validation (cv=3) within the training set only. Class imbalance was handled during training through model-specific weighting parameters rather than altering the global dataset.

### UCI:

Random Forest: class_weight="balanced"

XGBoost: scale_pos_weight = (negatives / positives) computed on the training fold

### Kaggle:

Random Forest: tuned via grid search (cv=3)

XGBoost: scale_pos_weight = (negatives / positives) computed on the training fold

No leakage rule: All encoders/scalers are fit only on X_train and applied to X_test.

## Reproducible split (reference code)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, stratify=y, random_state=42 )

Compute scale_pos_weight = (y_train==0).sum() / (y_train==1).sum() after the split.

Fit any StandardScaler or one-hot encoder on X_train only, then transform X_test.

## Engineered features (alignment for fair comparison)

UCI: AVG_PAY_DELAY = mean(PAY_0, PAY_2, …, PAY_6)

Kaggle: repayment_ratio = monthly_payment / total_loan, score_range = scoring_high - scoring_low

Categoricals: one-hot encode selected fields (e.g., UCI: EDUCATION, MARRIAGE; Kaggle: class, employer_type, industry) using an encoder fit on training only.

## Known biases & limits

Domain: UCI covers Taiwan credit-card customers; Kaggle covers one online lending platform in China.

Outcome prevalence: Default is the minority class; emphasis is on recall/F1 for target_default=1.

## Provenance & integrity checklist

Save the raw files exactly as downloaded in data/raw/, recording the download date, source URL, and licence or terms in docs/dmp.md. If the terms prohibit redistribution, do not share the raw data; instead, publish only the code, metadata, and derived non-sensitive outputs.

## Minimal rerun steps

Place the raw CSVs in data/raw/, then run preprocessing to generate data/processed/uci_clean_v1.csv and data/processed/kaggle_clean_v1.csv. Train with random_state=42, test_size=0.20, and stratify=y. For XGBoost, compute scale_pos_weight using only y_train during training. Export all metrics and plots to outputs/.
