# 📈 Comparative Assessment of Time Series Forecasting using TOTEM

This repository hosts the official implementation of the **Comparative Assessment of Time Series Forecasting using TOTEM (Tokenized Time Series Embeddings)**. Conducted under the **Samsung PRISM Program**, the project explores the performance of the TOTEM model across varied forecasting horizons and business domains, and compares it against classical and deep learning-based time series forecasting models.

---

## 🔍 Overview

Time series forecasting (TSF) plays a vital role in decision-making across industries like finance, supply chain, healthcare, and more. The recent rise of foundation models and tokenization-based techniques, such as **TOTEM**, provides a new perspective in time series analysis.

This project evaluates:
- The performance of **TOTEM** on diverse real-world time series datasets.
- Its adaptability to **different forecasting horizons**.
- Its **comparative strength** versus traditional TSF models (e.g., ARIMA, LSTM).
- The **efficacy of tokenization** in time series modeling.

---

## 🎯 Objectives

- 📦 Collect and preprocess time series data from multiple business domains.
- 📊 Perform exploratory data analysis (EDA) to identify intrinsic dataset properties.
- 🤖 Implement forecasting using TOTEM and baseline models.
- 📉 Assess and compare performance metrics across forecasting horizons.
- 🧪 Analyze the role of tokenization in TS forecasting.

---

## 🗂️ Repository Structure

📁 totem-tsf-analysis/ │ ├── data/ # Time series datasets ├── notebooks/ # Colab notebooks for analysis & modeling │ ├── 01_eda.ipynb │ ├── 02_totem_forecasting.ipynb │ ├── 03_baseline_models.ipynb │ └── 04_comparative_analysis.ipynb ├── results/ # Visualizations, metrics, comparison tables ├── requirements.txt # Dependencies └── README.md # Project documentation


---

## 🛠️ Tools & Technologies

- **Google Colab** for development
- **Python** (Pandas, Numpy, Scikit-learn, Matplotlib, etc.)
- **Deep Learning** (PyTorch / TensorFlow for LSTM, Transformers)
- **Time Series Libraries** (statsmodels, GluonTS, Darts)
- **TOTEM Model** (Custom or from existing implementation)

---

## 📅 Project Timeline

| Phase           | Milestone Description                                       |
|----------------|-------------------------------------------------------------|
| Month 1         | Dataset gathering, initial EDA, research question framing  |
| Month 2         | Forecasting using TOTEM, model tuning                      |
| Month 3-4       | Benchmarking vs classical models, horizon-based comparison |
| Month 5-6       | Analysis of results, final report, and presentation        |

---

## 📈 Sample Outputs (to be added)
- Forecast vs Actual Plots
- RMSE / MAE / MAPE tables
- Tokenization impact charts

---

## 📚 References

- [TOTEM Research Paper](https://arxiv.org/abs/2402.16412)
- [Time Series Foundation Models Overview](#)
- [GluonTS](https://ts.gluon.ai/)
- [Darts](https://github.com/unit8co/darts)

---

## 👥 Contributors

### 👨‍🎓 Student Researchers
- **Deben Kumar Jena**
- **Bishmaya Bhuyan**

### 👨‍🏫 Mentor
- **Prof. Aniket Shivam**

### 🏢 Samsung PRISM Mentors
- **Vikas Kumar Singh** – Senior Chief Engineer
- **Santanu Bhattacharjee** – Senior Chief Engineer
- **Pichili Sai Kishore Reddy** – Lead Engineer

---

## 🆔 Project Info

- **Worklet ID:** `25APBD02ITER`
- **Platform:** Samsung PRISM
- **Duration:** 6 Months
- **Focus Area:** AI, ML, Time Series Forecasting

---

## 📬 Contact

For queries, suggestions, or contributions:
- Open an issue on this repo
- Contact project mentors or contributors

---

> 🚀 This research is part of the Samsung PRISM initiative and is intended for academic and research use only.
