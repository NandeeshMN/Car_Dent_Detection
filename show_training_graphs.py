import pickle
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -----------------------------
# LOAD TRAINING HISTORY
# -----------------------------
history_path = os.path.join("models", "training_history.pkl")

if not os.path.exists(history_path):
    print("❌ training_history.pkl not found. Train the model first.")
    exit()

with open(history_path, "rb") as f:
    history = pickle.load(f)

# -----------------------------
# TKINTER WINDOW
# -----------------------------
root = tk.Tk()
root.title("Model Training Performance")
root.geometry("900x500")

fig, ax = plt.subplots(1, 2, figsize=(9, 4))

# Accuracy graph
ax[0].plot(history["accuracy"], label="Train Accuracy")
ax[0].plot(history["val_accuracy"], label="Validation Accuracy")
ax[0].set_title("Accuracy vs Epochs")
ax[0].legend()
ax[0].grid(True)

# Loss graph
ax[1].plot(history["loss"], label="Train Loss")
ax[1].plot(history["val_loss"], label="Validation Loss")
ax[1].set_title("Loss vs Epochs")
ax[1].legend()
ax[1].grid(True)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()

