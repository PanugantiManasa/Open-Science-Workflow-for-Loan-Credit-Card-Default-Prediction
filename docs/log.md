# Data Acquisition & Processing Log

This log records all raw dataset acquisitions and subsequent processing steps so the workflow is auditable and reproducible.

## 1. Raw Dataset Downloads

### [UCI Default of Credit Card Clients]

- **Local file**: `data/raw/uci_default_2025-08-11.csv`
- **Source URL**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Licence**: This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- **Download date**: 2025-07-20
- **Shape**: 30,000 rows × 25 columns

### [Kaggle Loan Default – train_public.csv]

- **Local file**: `data/raw/kaggle_train_public_2025-08-11.csv`
- **Source URL**: https://www.kaggle.com/datasets/yangrujun/customer-default-prediction
- **Licence**: License not specified — treated as _all rights reserved_.
- **Download date**: 2025-07-20
- **Shape**: 80,000 rows × 47 columns

## 2. Processing Steps

### for uci dataset

- **Input file**: `uci_default_2025-07-20.csv`
- **Output file**: `uci_clean_V1.csv`
- **Changes made**:
  - Renamed target column from `default.payment.next.month` → `default`
  - Created `avg_pay_delay` from PAY_0 to PAY_6
  - One-hot encoded `EDUCATION`, `MARRIAGE`
  - Dropped `ID`
- **Rows removed**: 0
- **Rows added**: 0

### for kaggle dataset

- **Input file**: `kaggle_train_public_2025-07-20.csv` (
- **Output file**: `kaggle_clean_v1.csv`
- **Changes made**:
  - Converted `work_year` text to numeric values
  - Imputed missing values in `pub_dero_bankrup` with median
  - Created `repayment_ratio` and `score_range`
  - One-hot encoded `class`, `employer_type`, `industry`
- **Rows removed**: 0
- **Rows added**: 0

---

## 3. Notes

- Raw files remain unchanged in `data/raw/` and are read-only.
- Processed files saved in `data/processed/`.
