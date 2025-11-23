import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----- Synthetic training data -----
epochs = np.arange(1, 13)
train_acc = np.linspace(0.45, 0.88, len(epochs))
val_acc = np.linspace(0.42, 0.83, len(epochs))
train_loss = np.linspace(1.8, 0.5, len(epochs))
val_loss = np.linspace(1.9, 0.6, len(epochs))

# ===== Accuracy plot =====
plt.figure()
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("accuracy.png")
plt.close()

# ===== Loss plot =====
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss.png")
plt.close()

# ===== Confusion matrix =====
true = np.random.randint(0, 6, 200)
pred = true + np.random.randint(0, 2, 200) - 1
pred = np.clip(pred, 0, 5)

cm = confusion_matrix(true, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("All graphs created: accuracy.png, loss.png, confusion_matrix.png")
