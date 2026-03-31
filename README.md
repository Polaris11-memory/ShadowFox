#  Boston House Price Predictor

> A complete end-to-end Machine Learning project — from raw data to an interactive live web app.

![ML](https://img.shields.io/badge/ML-Gradient%20Boosting-00e5ff?style=flat-square)
![R2](https://img.shields.io/badge/R²%20Score-0.9152-69f0ae?style=flat-square)
![RMSE](https://img.shields.io/badge/RMSE-%242.77k-7c6af7?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-f48fb1?style=flat-square)

---

##  Live Demo

> Deploy `web-app/` via GitHub Pages — see instructions below.

---


### Pipeline Overview

```
Raw CSV → Impute Missing Values → Clip Outliers → Train/Test Split
       → StandardScaler → Train 4 Models → Evaluate → GridSearchCV
       → Fine-tuned GBR → 8-Panel Visual Report
```

### Models Compared

| Model | MSE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 25.91 | 5.09 | 0.7145 |
| Ridge Regression | 25.90 | 5.09 | 0.7146 |
| Decision Tree | 30.91 | 5.56 | 0.6593 |
| Gradient Boosting | 8.22 | 2.87 | 0.9094 |
| **GBR Fine-tuned ** | **7.70** | **2.77** | **0.9152** |

### Best Model: Fine-tuned Gradient Boosting Regressor

| Parameter | Value |
|---|---|
| R² Score | **0.9152** (91.52% variance explained) |
| RMSE | **$2,774** |
| MAE | **$2,088** |
| CV-5 R² | 0.8503 ± 0.0641 |
| learning_rate | 0.1 |
| max_depth | 3 |
| n_estimators | 200 |
| subsample | 0.8 |

### Top Feature Importances

| Feature | Importance | Description |
|---|---|---|
| LSTAT | 38.54% | % lower-status population |
| RM | 37.01% | Average rooms per dwelling |
| DIS | 9.70% | Distance to employment centres |
| NOX | 4.23% | Pollution (nitric oxides) |
| CRIM | 3.20% | Per-capita crime rate |

### Run the ML Pipeline

```bash
# 1. Clone the repo
git clone https://github.com/Polaris11-memory/boston-house-predictor.git
cd boston-house-predictor/ml-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python boston_housing_prediction.py
```

This will:
- Preprocess the dataset (impute, clip outliers, scale)
- Train and evaluate 4 regression models
- Run GridSearchCV fine-tuning
- Save `boston_housing_report.png` — an 8-panel visual report

### Generated Report Preview

![Boston Housing Report](ml-model/boston_housing_report.png)

---

##  Web App — `web-app/`

A fully interactive, zero-dependency browser app with:

- **13 real-time sliders** — one per dataset feature
- **Instant price prediction** — updates as you move sliders
- **Price category badge** — Low / Affordable / Mid-Range / Premium / Luxury
- **Confidence meter** — based on training distribution proximity
- **Live feature driver chart** — shows top 6 prediction influences
- **Tooltips** — hover any feature for a plain-English description

### Run Locally

Just open `web-app/index.html` in any browser — no server or installation needed.

```bash
# From the repo root:
open web-app/index.html        # macOS
start web-app/index.html       # Windows
xdg-open web-app/index.html    # Linux
```

### Deploy via GitHub Pages (Free Hosting)

1. Push this repo to GitHub
2. Go to **Settings → Pages**
3. Set Source: **Deploy from a branch → main → / (root)**
4. Your app goes live at:
   ```
   https://Polaris11-memory.github.io/boston-house-predictor/web-app/
   ```

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Name | UCI Boston Housing Dataset |
| Year | 1978 |
| Samples | 506 |
| Features | 13 + 1 target (MEDV) |
| Missing Values | 20 per feature in 6 columns |

### Feature Reference

| Feature | Description |
|---|---|
| CRIM | Per-capita crime rate by town |
| ZN | % residential land zoned for large lots |
| INDUS | % non-retail business acres |
| CHAS | Charles River proximity (binary) |
| NOX | Nitric oxides concentration |
| RM | Average rooms per dwelling |
| AGE | % units built before 1940 |
| DIS | Distance to employment centres |
| RAD | Highway accessibility index |
| TAX | Property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio |
| B | Demographic index |
| LSTAT | % lower-status population |
| **MEDV** | **Median home value in $1000s ← Target** |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| ML Pipeline | Python, Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Web App | HTML5, CSS3, Vanilla JavaScript |
| Fonts | Google Fonts (Space Mono + Syne) |
| Hosting | GitHub Pages |

---

##  License

MIT License — free to use, modify, and distribute.

