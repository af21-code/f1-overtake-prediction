# 🏎️ F1 Overtake Dynamics
### A Machine Learning Approach for Predicting Position Changes

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview
This project aims to develop a **Machine Learning solution** to predict the probability of an overtaking maneuver in Formula 1 races based on telemetry data. The goal is to analyze lap-by-lap data to classify whether a driver will gain a position in the subsequent lap.

This repository contains the source code and the documentation for the Machine Learning course project (A.A. 2025-26).

## Objectives
* **Data Ingestion:** Automated extraction of telemetry data using the Ergast API and FastF1.
* **Preprocessing:** Handling missing values, outliers (e.g., Safety Car laps), and imbalanced classes (overtakes are rare events).
* **Feature Engineering:** Calculation of relative metrics (tire age difference, pace delta, gap).
* **Modelling:** Comparison of classification models (Logistic Regression, Random Forest, XGBoost).
* **Evaluation:** Performance analysis using Precision, Recall, and F1-Score.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **F1 Data Source:** FastF1 API
* **Machine Learning:** Scikit-Learn, Imbalanced-learn
* **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```text
f1-overtake-prediction/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and testing
├── src/                # Source code for the pipeline
├── .gitignore          # Python ignore file
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

Developed by Angelo Fusco for the Machine Learning Course, Prof. G. Polese & Prof. L. Caruccio.
