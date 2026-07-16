"""
train.py  —  Run once to build the model
Usage: python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.engine import generate_data, train

print("=" * 55)
print("   EduAI Pro — Model Training Pipeline")
print("=" * 55)

print("\n[1/3] Generating realistic student dataset …")
df = generate_data(n=3000)
os.makedirs("data", exist_ok=True)
df.to_csv("data/students.csv", index=False)
print(f"      {len(df):,} records  |  "
      f"Pass {(df.outcome=='Pass').mean()*100:.0f}%  "
      f"At-Risk {(df.outcome=='At-Risk').mean()*100:.0f}%  "
      f"Fail {(df.outcome=='Fail').mean()*100:.0f}%")

print("\n[2/3] Training Ensemble (RF + GBM + SVC + LR) …")
model, scaler, le, metrics = train(df, out_dir="saved_model")

print("\n[3/3] Artefacts saved to saved_model/")
print("\n" + "=" * 55)
print("  ✅  Done! Run the app:")
print("      streamlit run app/main.py")
print("=" * 55)
