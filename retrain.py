import os
import datetime
import subprocess

print("🚀 Starting retraining...")

# Run training script
subprocess.run(["python", "train/xgboost_model.py"], check=True)

print("✅ Retraining complete!")

# Optional: You can add Git commit logic here later