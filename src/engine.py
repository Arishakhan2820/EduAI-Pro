"""
EduAI Pro — Core ML Engine
Real-world feature engineering + ensemble model
"""

import numpy as np
import pandas as pd
import json, os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier,
    GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split,
    StratifiedKFold, cross_val_score)
from sklearn.metrics import (accuracy_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score)
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
FEATURES = [
    "study_hours_per_day",
    "sleep_hours",
    "attendance_pct",
    "prev_gpa",
    "assignment_score",
    "quiz_avg",
    "lab_score",
    "participation_score",
    "stress_level",
    "internet_hours",
    "part_time_job",
    "family_support",
    "commute_hours",
    "health_score",
    "extracurricular",
]

LABEL_COL  = "outcome"   # Pass / At-Risk / Fail
GPA_COL    = "final_gpa"

GRADE_MAP  = {
    (85, 101): ("A+", 4.0),
    (80,  85): ("A",  4.0),
    (75,  80): ("B+", 3.5),
    (70,  75): ("B",  3.0),
    (65,  70): ("C+", 2.5),
    (60,  65): ("C",  2.0),
    (50,  60): ("D",  1.0),
    ( 0,  50): ("F",  0.0),
}

# ─────────────────────────────────────────
# DATA GENERATION  (realistic distributions)
# ─────────────────────────────────────────
def generate_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def _beta(a, b, lo, hi, n=n):
        return rng.beta(a, b, n) * (hi - lo) + lo

    study     = _beta(3, 2,  1, 10).round(1)
    sleep     = _beta(4, 2,  4, 10).round(1)
    attend    = _beta(5, 2, 45,100).round(1)
    prev_gpa  = _beta(4, 2,  1,  4).round(2)
    assign    = _beta(4, 2, 40,100).round(1)
    quiz      = _beta(3, 2, 30,100).round(1)
    lab       = _beta(4, 2, 40,100).round(1)
    particip  = _beta(3, 3,  1, 10).round(1)
    stress    = _beta(2, 4,  1, 10).round(1)   # lower = better
    internet  = _beta(2, 4,  1, 10).round(1)   # lower = better
    job       = rng.choice([0, 1], n, p=[0.65, 0.35])
    fam_supp  = _beta(4, 2,  1, 10).round(1)
    commute   = _beta(2, 4,  0,  4).round(1)
    health    = _beta(4, 2,  1, 10).round(1)
    extra     = rng.choice([0, 1], n, p=[0.50, 0.50])

    # ── realistic final score formula ──────────────────────
    score = (
        study    * 4.0 +
        sleep    * 1.2 +
        attend   * 0.30 +
        prev_gpa * 8.0 +
        assign   * 0.18 +
        quiz     * 0.15 +
        lab      * 0.12 +
        particip * 1.0 +
        fam_supp * 0.8 +
        health   * 0.6 -
        stress   * 1.5 -
        internet * 0.8 -
        job      * 3.0 -
        commute  * 1.5 +
        extra    * 1.2 +
        rng.normal(0, 4, n)
    )

    # normalise to 0-100
    lo, hi  = np.percentile(score, [1, 99])
    final   = np.clip((score - lo) / (hi - lo) * 100, 0, 100).round(1)

    # ── 3-class outcome ────────────────────────────────────
    outcome = np.where(final >= 70, "Pass",
              np.where(final >= 50, "At-Risk", "Fail"))

    # ── grades / GPA ───────────────────────────────────────
    def _grade(m):
        for (lo, hi), (g, gp) in GRADE_MAP.items():
            if lo <= m < hi:
                return g, gp
        return "F", 0.0

    grades = [_grade(m)[0] for m in final]
    gpas   = [_grade(m)[1] for m in final]

    # ── demographics ───────────────────────────────────────
    cities    = rng.choice(["Karachi","Lahore","Islamabad",
                             "Peshawar","Quetta","Multan"], n,
                            p=[0.30,0.25,0.20,0.10,0.08,0.07])
    genders   = rng.choice(["Male","Female"], n, p=[0.52, 0.48])
    depts     = rng.choice(["CS","EE","BBA","Mech","Civil","Med"], n)
    semesters = rng.integers(1, 9, n)

    fn = ["Ali","Sara","Usman","Fatima","Bilal","Zara",
          "Ahmed","Hina","Kamran","Nadia","Hassan","Ayesha",
          "Omar","Sana","Tariq","Maria","Imran","Amna",
          "Faisal","Rabia"]
    ln = ["Khan","Ahmed","Ali","Hassan","Malik","Hussain",
          "Iqbal","Butt","Chaudhry","Sheikh"]
    names = [f"{rng.choice(fn)} {rng.choice(ln)}" for _ in range(n)]
    roll  = [f"STU-{str(i+1).zfill(4)}" for i in range(n)]

    df = pd.DataFrame({
        "roll_no":              roll,
        "name":                 names,
        "gender":               genders,
        "city":                 cities,
        "department":           depts,
        "semester":             semesters,
        "study_hours_per_day":  study,
        "sleep_hours":          sleep,
        "attendance_pct":       attend,
        "prev_gpa":             prev_gpa,
        "assignment_score":     assign,
        "quiz_avg":             quiz,
        "lab_score":            lab,
        "participation_score":  particip,
        "stress_level":         stress,
        "internet_hours":       internet,
        "part_time_job":        job,
        "family_support":       fam_supp,
        "commute_hours":        commute,
        "health_score":         health,
        "extracurricular":      extra,
        "final_score":          final,
        "grade":                grades,
        "final_gpa":            gpas,
        "outcome":              outcome,
    })
    return df


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # composite indices
    d["academic_index"] = (
        d["study_hours_per_day"] * 0.25 +
        d["assignment_score"]    * 0.20 +
        d["quiz_avg"]            * 0.20 +
        d["lab_score"]           * 0.15 +
        d["attendance_pct"]      * 0.20 / 10
    ).round(3)

    d["wellness_index"] = (
        d["sleep_hours"]   * 0.30 +
        d["health_score"]  * 0.40 +
        (10 - d["stress_level"]) * 0.30
    ).round(3)

    d["support_index"] = (
        d["family_support"]    * 0.50 +
        d["participation_score"] * 0.30 +
        d["extracurricular"]   * 3.0 * 0.20
    ).round(3)

    d["risk_index"] = (
        d["stress_level"]   * 0.30 +
        d["internet_hours"] * 0.25 +
        d["part_time_job"]  * 4.0 * 0.25 +
        d["commute_hours"]  * 2.0 * 0.20
    ).round(3)

    d["study_efficiency"] = (
        d["assignment_score"] /
        d["study_hours_per_day"].clip(lower=0.1)
    ).round(3)

    d["sleep_quality_flag"] = (d["sleep_hours"] < 6).astype(int)
    d["low_attend_flag"]    = (d["attendance_pct"] < 75).astype(int)
    d["high_stress_flag"]   = (d["stress_level"] > 7).astype(int)
    d["gpa_risk_flag"]      = (d["prev_gpa"] < 2.0).astype(int)

    return d


ENG_FEATURES = FEATURES + [
    "academic_index", "wellness_index", "support_index",
    "risk_index", "study_efficiency",
    "sleep_quality_flag", "low_attend_flag",
    "high_stress_flag", "gpa_risk_flag",
]


# ─────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────
def train(df: pd.DataFrame, out_dir: str = "saved_model"):
    os.makedirs(out_dir, exist_ok=True)

    df_eng = engineer_features(df)

    X = df_eng[ENG_FEATURES]
    le = LabelEncoder()
    y  = le.fit_transform(df_eng[LABEL_COL])   # 0=At-Risk,1=Fail,2=Pass

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── individual learners ────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=300, max_depth=10,
                                  min_samples_leaf=4, random_state=42,
                                  n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                      learning_rate=0.08,
                                      random_state=42)
    svc = CalibratedClassifierCV(SVC(kernel="rbf", C=2,
                                      probability=False,
                                      random_state=42))
    lr  = LogisticRegression(C=1.0, max_iter=2000,
                              random_state=42)

    # ── soft-voting ensemble ───────────────────────────────
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb),
                    ("svc", svc), ("lr", lr)],
        voting="soft", n_jobs=-1
    )
    ensemble.fit(X_tr_s, y_tr)

    # ── evaluation ─────────────────────────────────────────
    y_pr   = ensemble.predict(X_te_s)
    y_prob = ensemble.predict_proba(X_te_s)

    cv = cross_val_score(ensemble, X_tr_s, y_tr,
                          cv=StratifiedKFold(5),
                          scoring="accuracy", n_jobs=-1)

    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pr), 4),
        "f1_macro":  round(f1_score(y_te, y_pr, average="macro"), 4),
        "roc_auc":   round(roc_auc_score(y_te, y_prob,
                            multi_class="ovr", average="macro"), 4),
        "cv_mean":   round(cv.mean(), 4),
        "cv_std":    round(cv.std(), 4),
        "n_train":   len(X_tr),
        "n_test":    len(X_te),
        "classes":   list(le.classes_),
        "features":  ENG_FEATURES,
        "n_features": len(ENG_FEATURES),
    }

    # ── feature importance from RF leg ─────────────────────
    rf_leg = ensemble.estimators_[0]
    fi = dict(zip(ENG_FEATURES,
                  rf_leg.feature_importances_.round(4)))
    metrics["feature_importances"] = fi

    # ── confusion matrix ───────────────────────────────────
    cm = confusion_matrix(y_te, y_pr)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_report"] = classification_report(
        y_te, y_pr,
        target_names=le.classes_,
        output_dict=True
    )

    # ── save artefacts ─────────────────────────────────────
    joblib.dump(ensemble, f"{out_dir}/model.pkl")
    joblib.dump(scaler,   f"{out_dir}/scaler.pkl")
    joblib.dump(le,       f"{out_dir}/label_encoder.pkl")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"✅  F1 (macro): {metrics['f1_macro']*100:.2f}%")
    print(f"✅  ROC-AUC   : {metrics['roc_auc']*100:.2f}%")
    print(f"✅  CV 5-fold : {metrics['cv_mean']*100:.2f}% "
          f"± {metrics['cv_std']*100:.2f}%")
    return ensemble, scaler, le, metrics


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
def load_artefacts(out_dir: str = "saved_model"):
    model = joblib.load(f"{out_dir}/model.pkl")
    scaler= joblib.load(f"{out_dir}/scaler.pkl")
    le    = joblib.load(f"{out_dir}/label_encoder.pkl")
    with open(f"{out_dir}/metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, le, metrics


def predict_one(raw: dict, model, scaler, le) -> dict:
    row = pd.DataFrame([raw])
    row = engineer_features(row)
    X   = row[ENG_FEATURES]
    Xs  = scaler.transform(X)

    cls_idx  = model.predict(Xs)[0]
    probs    = model.predict_proba(Xs)[0]
    label    = le.inverse_transform([cls_idx])[0]
    prob_map = {c: round(float(p)*100, 1)
                for c, p in zip(le.classes_, probs)}

    confidence = max(probs) * 100

    # ── risk score 0-100 (lower = safer) ───────────────────
    fail_prob    = prob_map.get("Fail", 0)
    at_risk_prob = prob_map.get("At-Risk", 0)
    risk_score   = round(fail_prob * 0.7 + at_risk_prob * 0.3, 1)

    # ── personalised recommendations ───────────────────────
    recs = _recommend(raw, label, risk_score)

    # ── estimated grade ────────────────────────────────────
    p    = prob_map.get("Pass", 0) / 100
    ar   = prob_map.get("At-Risk", 0) / 100
    est  = p * 80 + ar * 58 + (1 - p - ar) * 35
    grade, gpa = _score_to_grade(est)

    return {
        "outcome":      label,
        "probabilities": prob_map,
        "confidence":   round(confidence, 1),
        "risk_score":   risk_score,
        "est_score":    round(est, 1),
        "est_grade":    grade,
        "est_gpa":      gpa,
        "recommendations": recs,
    }


def predict_batch(df: pd.DataFrame, model, scaler, le) -> pd.DataFrame:
    eng = engineer_features(df)
    X   = eng[ENG_FEATURES]
    Xs  = scaler.transform(X)

    preds = model.predict(Xs)
    probs = model.predict_proba(Xs)

    out              = df.copy()
    out["outcome"]   = le.inverse_transform(preds)
    for i, cls in enumerate(le.classes_):
        out[f"prob_{cls.lower().replace('-','_')}"] = (
            probs[:, i] * 100).round(1)
    out["risk_score"] = (
        out.get("prob_fail", 0) * 0.7 +
        out.get("prob_at_risk", 0) * 0.3
    ).round(1)
    return out


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _score_to_grade(score):
    for (lo, hi), (g, gp) in GRADE_MAP.items():
        if lo <= score < hi:
            return g, gp
    return "F", 0.0


def _recommend(d: dict, outcome: str, risk: float) -> list:
    tips = []

    if d.get("attendance_pct", 100) < 75:
        tips.append({"icon": "🏫", "level": "danger",
            "title": "Critical: Low Attendance",
            "body":  f"Attendance is {d['attendance_pct']}% — "
                     "below the 75% minimum. Missing classes "
                     "directly lowers exam scores by up to 15 marks."})

    if d.get("study_hours_per_day", 0) < 3:
        tips.append({"icon": "📚", "level": "danger",
            "title": "Insufficient Study Time",
            "body":  "Less than 3 hours daily study is insufficient. "
                     "Target 5–7 hours with Pomodoro technique."})

    if d.get("sleep_hours", 8) < 6:
        tips.append({"icon": "😴", "level": "warning",
            "title": "Sleep Deprivation Detected",
            "body":  "Below 6 hours sleep impairs memory consolidation "
                     "by 40%. Aim for 7–8 hours consistently."})

    if d.get("stress_level", 0) > 7:
        tips.append({"icon": "🧘", "level": "warning",
            "title": "High Stress Level",
            "body":  "Chronic stress degrades cognitive performance. "
                     "Try 10-min daily mindfulness or exercise."})

    if d.get("prev_gpa", 4) < 2.0:
        tips.append({"icon": "📈", "level": "danger",
            "title": "Low Historical GPA",
            "body":  "Previous GPA < 2.0 strongly predicts difficulty. "
                     "Schedule weekly sessions with academic advisor."})

    if d.get("assignment_score", 100) < 60:
        tips.append({"icon": "📝", "level": "warning",
            "title": "Assignment Performance Low",
            "body":  "Assignments < 60% signals concept gaps. "
                     "Revisit weak topics using past papers."})

    if d.get("internet_hours", 0) > 6:
        tips.append({"icon": "📵", "level": "info",
            "title": "Excessive Screen Time",
            "body":  "6+ hours daily internet reduces focus. "
                     "Use app blockers during study windows."})

    if d.get("part_time_job") == 1:
        tips.append({"icon": "💼", "level": "info",
            "title": "Work-Study Balance",
            "body":  "Part-time work increases dropout risk by 22%. "
                     "Ensure ≥5 dedicated study hours despite work."})

    if not tips:
        tips.append({"icon": "🌟", "level": "success",
            "title": "Excellent Profile!",
            "body":  "All indicators are healthy. "
                     "Maintain consistency and aim for A+."})

    return tips
