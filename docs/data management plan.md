# Data Management Plan

## Project: Comparative Open-Data Credit Default Prediction

## summary

Two open datasets (UCI Credit Card Default, Kaggle Loan Default) are used separately to build reproducible ML pipelines predicting default. All handling follows FAIR principles: raw data is immutable, transformations documented, and non-restricted outputs/code shared openly.

## Data Sources

UCI: 30k clients, Taiwan – Target: default. [URL](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) , Licence: This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

Kaggle: ~10k loans – Target: isDefault. [URL](https://www.kaggle.com/datasets/yangrujun/customer-default-prediction) + Licence: License not specified — treated as _all rights reserved_.

## Storage & Structure

data/raw/ # originals (read-only, with hashes) data/processed/ # analysis-ready tables (timestamped) docs/ # DMP, logs code/ # scripts figures/ # metrics, figures

Backups: 3–2–1 rule (local, remote mirror, Zenodo).

## Processing Steps

Drop IDs, clean categories, engineer features (AVG_PAY_DELAY, repayment_ratio).

Stratified 80/20 split with fixed seed.

Impute missing values (Kaggle only).

Encode categoricals, scale where needed.

Save split indices and versioned outputs.

## Access & Licensing

Original licences will be retained with the datasets, and no redistribution will occur if restrictions apply. Only the code, metadata, and permitted outputs will be published.

## Preservation & Sharing

At the project’s conclusion, a public repository will be released and archived on Zenodo with a DOI, containing the README, LICENSE, documentation, and the outputs/ directory

## Risks & Mitigation

Licence uncertainty will be resolved before any data sharing. To address class imbalance, we used stratified train-test splits and applied model weighting techniques.
