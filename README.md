# Loan Default Prediction — Open Data & Open Science

## Overview

ChatGPT said: This project applies open science principles to develop predictive models for loan and credit-card default risk using two public datasets—UCI Default of Credit Card Clients and Kaggle Loan Default Prediction. In line with the FAIR principles (Findable, Accessible, Interoperable, Reusable), all data transformations are documented, and derived non-sensitive outputs are published on Zenodo for open access and citation.

## Project Objectives

Compare modelling approaches on two distinct default-prediction datasets.

Build transparent, reproducible preprocessing and modelling workflows.

Demonstrate open-data handling, documentation, and publication.

## Datasets

UCI Default of Credit Card Clients

Source: UCI Machine Learning Repository

Target variable: default payment next month

Rows: 30,000 | Features: 26

Kaggle Loan Default Prediction

Source: Kaggle Competition

Target variable: isDefault

Rows: 10,000 (subset for this project) | Features: 36

Raw files are stored in data/raw/ (read-only). Cleaned, analysis-ready datasets are in data/processed/.

## Data Management

Details are outlined in docs/data management plan.md, where all raw files are stored using the source name and download date. The docs/log.md file records dataset URLs, licences, file sizes, and row/column counts. Cleaning procedures and imputation logic are documented in the data dictionary within the notebooks, and licensing information is preserved alongside each dataset..

## Processing Workflow

We preprocessed both datasets in advance to avoid redundant processing during repeated model runs. For the UCI dataset, we dropped non-predictive IDs, created the AVG_PAY_DELAY feature, and one-hot encoded EDUCATION and MARRIAGE. For the Kaggle dataset, we cleaned the work_year field, imputed missing values in pub_dero_bankrup and f0–f4, created repayment_ratio and score_range, and one-hot encoded categorical variables. The processed files are stored in data/processed/ and are loaded directly in the \*\_fast_run.py scripts.

## Analysis & Results

Statistical summaries and visualisations are stored in figures/ and outputs/, including class distribution bar charts, feature histograms, correlation heatmaps, and summary statistics tables. Detailed explanations of these analyses are provided in the Data Analysis section of the report.

## Modelling

For each dataset, we applied three models: Logistic Regression (with StandardScaler), Random Forest (with grid search hyperparameter tuning), and XGBoost (with grid search tuning and scale_pos_weight adjustment). Model evaluation included confusion matrices, classification reports (Precision, Recall, F1-score), ROC curves, feature importance plots, and SHAP explainability analysis performed on a 100-row sample.

## Quick Start

pip install -r docs/requirements.txt Run Fast Pipeline

### For UCI dataset

python src/uci_fast_run.py

### For Kaggle dataset

python src/kaggle_fast_run.py

## Citation

This project’s cleaned datasets, figures, and code are published on Zenodo: DOI: 10.xxxx/zenodo.xxxxxxx
