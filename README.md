# 🎓 EduAI Pro — Academic Performance Intelligence Platform

> **End-to-end ML system** for early at-risk detection, GPA forecasting, and personalised student interventions — built with a **4-learner soft-voting ensemble** and a production-grade Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-f7931e?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26-ff4b4b?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3f4f75?logo=plotly&logoColor=white)

---

## ✨ What Makes This Different

| Feature | Details |
|---------|---------|
| **Real ML** | 4-learner Voting Ensemble (RF + GBM + SVC + LR) |
| **3-class Output** | Pass / At-Risk / Fail — not just pass/fail |
| **Feature Engineering** | 9 composite indices from 15 raw features |
| **ROC-AUC ~89%** | Evaluated with 5-fold stratified CV |
| **Professional UI** | Dark-theme, Plotly charts, gauge meter, badges |
| **Batch Analysis** | Upload CSV → download enriched predictions |
| **AI Recommendations** | Evidence-based, personalised per student |
| **Interactive Dashboard** | Filterable EDA with correlation heatmap |

---

## 📁 Project Structure

```
eduai_pro/
├── src/
│   └── engine.py          # Data gen + feature eng + train + predict
├── app/
│   └── main.py            # 5-page Streamlit application
├── data/
│   └── students.csv       # Auto-generated (3,000 records)
├── saved_model/
│   ├── model.pkl          # Trained ensemble
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── metrics.json
├── train.py               # One-command training script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (runs in ~60 seconds)
python train.py

# 3. Launch the app
streamlit run app/main.py
```

---

## 🧠 ML Architecture

```
Raw Features (15)
      ↓
Feature Engineering  →  +9 composite indices (24 total)
      ↓
StandardScaler
      ↓
┌──────────────────────────────┐
│  Soft-Voting Ensemble        │
│  ├── RandomForest (300 trees)│
│  ├── GradientBoosting (200)  │
│  ├── SVC (RBF, calibrated)   │
│  └── LogisticRegression      │
└──────────────────────────────┘
      ↓
3-class Output: Pass / At-Risk / Fail
      ↓
Probability calibration + Risk Score
      ↓
Personalised Recommendations
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC (macro OvR) | **~89.5%** |
| F1 Macro | **~73%** |
| CV Accuracy (5-fold) | **~75% ± 2%** |
| Training samples | 2,400 |
| Test samples | 600 |

---

## 🖥️ App Pages

1. **Overview** — KPI cards, feature importance, confusion matrix
2. **Predict Student** — Form input → outcome + gauge + recommendations
3. **Batch Analysis** — CSV upload → class-wide predictions + download
4. **Analytics Dashboard** — Filterable EDA, correlation heatmap, boxplots
5. **Model Insights** — Per-class report, model card, raw metrics JSON

---

## 📄 License
MIT
