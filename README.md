# UCL Player Goal Predictor

A machine learning project to classify UEFA Champions League players as **high scorers** or **low scorers** using attacking and shooting statistics.

This repository contains:
- data files (`attacking.csv`, `attempts.csv`, `goals.csv`)
- the full notebook workflow (`notebooks_UCL_Goal_Predictor_UCL_Player_GoalPredictor.ipynb`)
- poster artifacts (`UCL_Goal_Prediction_E-Poster.html`, `e-poster.html`)
- supporting visual outputs (`high_scorer_distribution.png`, `goals_vs_attempts.png`, `feature_importance.png`)

---

## Project Objective

Predict whether a player is likely to be a high goal scorer using pre-goal indicators such as:
- shot volume (`total_attempts`)
- shot precision (`on_target`, engineered `shot_accuracy`)
- attacking behavior (`assists`, `dribbles`, `offsides`, etc.)
- match participation features

The prediction task is formulated as **binary classification**:
- `1` = High Scorer (goals >= 70th percentile threshold)
- `0` = Low Scorer

---

## Dataset

Source: [Kaggle - UCL 2021/22 UEFA Champions League Player Stats](https://www.kaggle.com/datasets/azminetoushikwasi/ucl-202122-uefa-champions-league)

### Files Used
- `attacking.csv` (~176 rows)
- `attempts.csv` (~546 rows)
- `goals.csv` (~183 rows)

After merging and cleaning, modeling is performed on approximately **546 players**.

---

## ML Workflow (Notebook)

Implemented in `notebooks_UCL_Goal_Predictor_UCL_Player_GoalPredictor.ipynb`.

### 1) Data Preparation
- normalize column names
- standardize `player_name` keys
- merge datasets (attempts + goals + attacking)
- remove duplicate/redundant columns
- impute missing values

### 2) Feature Engineering
- create `shot_accuracy = on_target / max(total_attempts, 1)`
- encode required categorical fields

### 3) Leakage Prevention
Goal-revealing columns are removed from model inputs (for example: direct goal outputs and conversion features) to avoid inflated/invalid performance.

### 4) Train/Test Split
- 80/20 split
- `random_state=42`
- scaling applied for Logistic Regression pipeline

### 5) Models Compared
- Logistic Regression
- Random Forest
- XGBoost

---

## Results Summary

From notebook outputs:
- Logistic Regression: **0.7818**
- Random Forest: **0.9000**
- XGBoost: **0.9182**

Poster versions in this repository present the same comparative conclusion that **tree-based models outperform Logistic Regression**, with Random Forest and XGBoost as top performers.

---

## Visualizations Included

- `high_scorer_distribution.png` - class distribution overview
- `goals_vs_attempts.png` - goals vs attempt correlation
- `feature_importance.png` - top feature importance view

These are embedded in both poster HTML files.

---

## Poster Files

### `UCL_Goal_Prediction_E-Poster.html`
- full rich poster version
- includes method, results, visual insights, feature importance, conclusion, and references

### `e-poster.html`
- compact single-page A4 landscape version optimized for PDF export
- includes visual insights and feature importance images with constrained layout for print fit

---

## How to Run the Notebook

## 1) Create and activate environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2) Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## 3) Start Jupyter

```bash
jupyter notebook
```

Open:
`notebooks_UCL_Goal_Predictor_UCL_Player_GoalPredictor.ipynb`

---

## Export Poster to PDF (Single Page)

Use `e-poster.html`:
1. Open in a Chromium-based browser (Chrome/Edge).
2. Press `Ctrl + P`.
3. Choose:
   - Paper size: **A4**
   - Orientation: **Landscape**
   - Margins: **None**
   - Scale: **100%** (or fit to page if needed)
4. Save as PDF.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- XGBoost
- HTML/CSS (poster design and PDF-ready layout)

---

## Author

**Aryan Dev Tyagi**  
Course: **BCAI-601-MLT (Machine Learning Techniques), 2025-26**

