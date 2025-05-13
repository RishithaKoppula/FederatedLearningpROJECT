# Federated Learning using FedAvg

## 📌 Overview

This project implements a Federated Learning (FL) framework using the **Federated Averaging (FedAvg)** algorithm. The goal is to demonstrate privacy-preserving machine learning where models are trained collaboratively across decentralized datasets, without raw data ever leaving the local clients.

We apply this setup to two datasets:
- A **synthetic dataset** of 2D Gaussian distributions to simulate non-IID client environments
- A **real-world property dataset** from New York State to demonstrate practical use cases in regression

This work addresses how FedAvg behaves under heterogeneous conditions, and how preprocessing, tuning, and centralized baselines impact federated model convergence.

---

## 🧠 What This Project Does

- **Synthetic Client Simulation**: Each client receives samples from a unique 2D Gaussian, mimicking real-world data drift and imbalance.
- **Real Dataset Partitioning**: NYS property dataset is partitioned by a simulated `CLIENT_ID` to form multiple clients with structured, tabular data.
- **Local SGD Training**: Each client trains locally using `SGDRegressor` for one epoch per communication round.
- **FedAvg Aggregation**: The global model is updated via weighted averaging of client weights.
- **Centralized Baseline**: A separate regression model is trained on the entire dataset to benchmark FedAvg convergence.
- **MSE Tracking**: Global MSE, local average MSE, and baseline MSE are plotted across 30 communication rounds.

---

## 📚 Why This Matters

Federated Learning is vital in fields like healthcare, finance, and personal devices — where raw data cannot be shared. Our implementation shows how:
- Standardized preprocessing enables convergence
- FedAvg can approach centralized performance
- Evaluation against a true centralized baseline is critical for analysis

This project highlights the importance of properly tuning learning rate, scaling data, and using reliable metrics in federated setups.

---

## 🎓 Academic Context

This project was completed as part of **AMAT 593 – Practical Methods in Machine Learning** at **SUNY Albany**. It was presented in the **departmental poster showcase**, focusing on:
- FedAvg behavior under non-IID synthetic and real-world conditions
- Comparison of federated vs. centralized learning outcomes
- Visualization of global vs. local MSE over training rounds

---

## 🔬 Technologies Used

- **Python 3.10**
- **scikit-learn** – for `SGDRegressor`, `StandardScaler`, and metrics
- **Pandas** – for data preprocessing and grouping by client
- **NumPy** – for manual aggregation of model weights
- **Matplotlib** & **Seaborn** – for MSE visualization

---

## 📈 Sample Use Cases

The architecture can be extended to:
- Predictive modeling across hospitals without sharing medical records
- House price prediction using private real estate data from multiple brokers
- Cross-device personalization models where user privacy is essential

---

This repository is designed for those looking to get a hands-on, interpretable start with Federated Learning using real and simulated data, with direct visualization of key metrics like MSE.
