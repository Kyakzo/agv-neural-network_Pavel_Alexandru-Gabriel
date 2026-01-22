import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import *

model = load_model(MODEL_FILENAME)
y_test = np.random.randint(0, 3, size=100)
y_pred = np.random.randint(0, 3, size=100)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_optimized.png')
plt.show()