import pandas as pd
import matplotlib.pyplot as plt

# ==== 1. Load CSV ====
# change this to your actual file name
log_path = "logs/training_log_20251118_150545_firstgoodone.csv"
df = pd.read_csv(log_path)

epochs = df["Epoch"]

# ==== 2. Train vs Val Accuracy ====
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["Train Acc"], label="Train Acc")
plt.plot(epochs, df["Val Acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ==== 3. Training Loss ====
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["Train Loss"], label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ==== 4. Train vs Val MCC ====
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["Train MCC"], label="Train MCC")
plt.plot(epochs, df["Val MCC"], label="Val MCC")
plt.xlabel("Epoch")
plt.ylabel("MCC")
plt.title("Train vs Validation MCC")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ==== 5. Validation AUC ====
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["Val AUC"], label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation AUC")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ==== 6. TP / TN / FP / FN over epochs ====
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["TP"], label="TP")
plt.plot(epochs, df["TN"], label="TN")
plt.plot(epochs, df["FP"], label="FP")
plt.plot(epochs, df["FN"], label="FN")
plt.xlabel("Epoch")
plt.ylabel("Count")
plt.title("Confusion Matrix Counts over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ==== 7. Epoch time ====
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["Epoch Time (s)"], label="Epoch Time")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Epoch Duration")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()