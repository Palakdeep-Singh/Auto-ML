<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/PRs-Welcome-6366f1?style=for-the-badge"/>
<img src="https://img.shields.io/github/issues/Palakdeep-Singh/Auto-ML?style=for-the-badge&color=f97316"/>
<img src="https://img.shields.io/github/stars/Palakdeep-Singh/Auto-ML?style=for-the-badge&color=facc15"/>

<br/><br/>

# 🤖 AutoML Studio

### A no-code, browser-based Automated Machine Learning platform
### — from raw CSV to a trained, downloadable model in minutes.

<br/>

> Upload → Auto-clean → Select target → Engineer features → Train → Analyse → Download your pipeline.
> **Zero Python knowledge required.**

<br/>

[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp; [✨ Features](#-features) &nbsp;·&nbsp; [🗺️ App Flow](#%EF%B8%8F-app-flow) &nbsp;·&nbsp; [🧠 Models](#-supported-models) &nbsp;·&nbsp; [🤝 Contributing](#-contributing) &nbsp;·&nbsp; [📄 License](#-license)

</div>

---

## ✨ Features

| | What it does |
|---|---|
| 📂 **CSV Upload** | Upload any CSV with instant validation and preview |
| 🔍 **Auto EDA** | Distributions, correlations, missing-value heatmaps — generated automatically |
| 🧹 **Smart Auto-Clean** | Fixes missing values, type mismatches, and near-empty columns in one click |
| 🎯 **Target Selection** | Pick your target column; task type (regression / classification) is auto-detected |
| ⚙️ **Feature Engineering** | Async background generation — polynomial, interaction, and statistical transforms |
| 🔎 **Feature Selection** | Importance-based ranking so you train only on what matters |
| 🏋️ **Model Training** | Multiple models trained simultaneously with a live progress dashboard |
| 🔧 **Hyperparameter Tuning** | Grid Search & Randomized Search — full auto or manual-override |
| 📊 **Results & Analysis** | Confusion matrices, residual plots, feature importance, cross-val scores |
| 📦 **Export** | Download the trained sklearn `Pipeline` as `.pkl` or processed data as `.csv` |
| 📋 **Report** | One-click downloadable HTML training report |
| 📖 **About & Guide** | Built-in step-by-step user guide |

---

## 🗺️ App Flow

```
  Step 1 │ Upload CSV
         │   Validates file → stores raw DataFrame

  Step 2 │ Data Overview  ← runs automatically after upload
         │   EDA · shape · dtypes · missing values · correlation heatmap
         │   Auto-cleans: drops near-empty columns, fixes types, imputes obvious nulls

  Step 3 │ Target Selection  ← must happen before Feature Engineering
         │   Pick target column
         │   Auto-detects: Regression / Classification / Zero-Inflated

  Step 4 │ Feature Engineering  ← runs async in background
         │   Polynomial · interaction · statistical transforms

  Step 5 │ Feature Selection
         │   Importance ranking · user selects final feature set

  Step 6 │ Model Training
         │   Select models (auto or manual) · configure hyperparameters
         │   Live progress bar · per-model metrics stream in real-time

  Step 7 │ Results & Analysis
         │   Side-by-side model comparison · plots · best-model highlight

  Step 8 │ Download
         │   Trained pipeline (.pkl) · processed dataset (.csv) · HTML report
```

---

## 🧠 Supported Models

### 📉 Regression
| Model | Tunable |
|---|---|
| Linear Regression | ✅ |
| Ridge Regression | ✅ |
| Lasso Regression | ✅ |
| Random Forest Regressor | ✅ |
| Gradient Boosting Regressor | ✅ |
| Support Vector Regressor | ✅ |
| KNN Regressor | ✅ |
| AdaBoost Regressor | ✅ |

### 🏷️ Classification
| Model | Tunable |
|---|---|
| Logistic Regression | ✅ |
| Random Forest Classifier | ✅ |
| Gradient Boosting Classifier | ✅ |
| Support Vector Machine | ✅ |
| KNN Classifier | ✅ |
| AdaBoost Classifier | ✅ |

---

## 🚀 Quick Start

### 1 — Clone

```bash
git clone https://github.com/Palakdeep-Singh/Auto-ML.git
cd Auto-ML
```

### 2 — Virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** and click **Upload Data** to begin. 🎉

---

## 📁 Project Structure

```
Auto-ML/
├── app.py                    # Flask routes & entry point
├── utility.py                # Core ML logic — cleaning, training, FE, selection
├── models_registry.py        # All models + hyperparameter configurations
├── state.py                  # Shared runtime state (DATASTORE, TRAINING_STATUS)
├── split.py / split_ast.py / split_lines.py   # Dataset splitting helpers
├── requirements.txt
│
├── static/
│   ├── custom.css
│   ├── customer_churn.csv    # Sample dataset
│   ├── heart_disease.csv     # Sample dataset
│   └── house_prices.csv      # Sample dataset
│
└── templates/
    ├── base.html             # Master layout (sidebar, topbar, CSS vars)
    ├── index.html
    ├── upload.html
    ├── dataset_overview.html
    ├── target.html
    ├── fe_auto.html
    ├── feature_selection.html
    ├── model_training.html   # Training dashboard — most active development area
    ├── model_comparison.html
    ├── results.html
    ├── report.html
    └── about.html
```

---

## 🌐 Internal API

| Endpoint | Method | Description |
|---|---|---|
| `/api/training_status` | GET | Live training progress & per-model metrics |
| `/api/processing_status` | GET | Data-processing pipeline status |
| `/api/fe_progress` | GET | Feature engineering async progress |
| `/api/fe_summary` | GET | Summary of engineered features |
| `/api/column_stats/<col>` | GET | Per-column statistics |
| `/api/reset_progress` | POST | Reset training progress state |

---

## 📦 Dependencies

```
astunparse==1.6.3
Flask==3.1.3
joblib==1.5.2
loky==3.5.6
matplotlib==3.10.8
numpy==2.4.2
pandas==3.0.1
scikit_learn==1.8.0
seaborn==0.13.2
```

---

## 🤝 Contributing

Contributions are welcome! Read **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** for the full guide and **[`ISSUES.md`](./ISSUES.md)** for tracked bugs open for contribution.

```bash
git checkout -b fix/your-fix-name
git commit -m "fix(ui): describe your change"
git push origin fix/your-fix-name
# → open a Pull Request
```

New? Look for [`good first issue`](https://github.com/Palakdeep-Singh/Auto-ML/issues?q=label%3A%22good+first+issue%22) labels.

---

## 👥 Contributors

See [`CONTRIBUTORS.md`](./CONTRIBUTORS.md).

---

## 📄 License

MIT — see [`LICENSE`](./LICENSE).

---

<div align="center">

Made with ❤️ by [Palakdeep Singh](https://github.com/Palakdeep-Singh) &nbsp;·&nbsp; ⭐ Star the repo if it helps you!

</div>
