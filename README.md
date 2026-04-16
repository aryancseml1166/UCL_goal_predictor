# ⚽ UCL Player Goal Predictor

A Machine Learning project that predicts whether a player in the UEFA Champions League (UCL) will be a **high goal scorer** using attacking and shooting statistics.

---

## 📌 Overview

This project uses machine learning classification techniques to identify high-scoring players based on **pre-game observable features** such as shot attempts, accuracy, and match participation.

Instead of directly using goal-related data (which leads to data leakage), the model learns patterns from **attacking behaviour and performance metrics**.

---

## 🎯 Problem Statement

The objective is to predict whether a player will be a **high scorer** using only:

- Shooting statistics  
- Attacking metrics  
- Match participation  

### 🧠 Task Type
**Binary Classification**
- `1` → High Scorer (Top 30% players based on goals)  
- `0` → Low Scorer  

---

## 📂 Dataset Description

Dataset Source: Kaggle – *UCL 2021/22 Player Statistics*

### Files Used:
- `attacking.csv`
- `attempts.csv`
- `goals.csv`

### 📊 Final Dataset:
- **546 players**
- **11 features**
- **1 target variable**

### Target Distribution:
- **Low Scorer (0):** ~79%  
- **High Scorer (1):** ~21%  

> The dataset is imbalanced due to a large number of players with low or zero goals.

---

## ⚙️ Data Preprocessing

The dataset was cleaned and prepared using a structured pipeline:

### Steps:
1. Column normalization (lowercase, remove spaces)  
2. Player name standardization for merging  
3. Merging datasets using `player_name`  
4. Removing duplicate and irrelevant columns  
5. Handling missing values:
   - Numeric → mean  
   - Goals → 0  
6. One-hot encoding for categorical features (position)  
7. Feature engineering:

```python
shot_accuracy = on_target / max(total_attempts, 1)
