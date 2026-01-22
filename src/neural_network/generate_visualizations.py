import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import *

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_learning_curves():
    history_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'training_history.csv')
    if not os.path.exists(history_path):
        print("Training history not found. Skipping learning curves.")
        return

    df = pd.read_csv(history_path)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Validation Loss')
    plt.title('Model Loss Evolution (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if 'accuracy' in df.columns:
        plt.plot(df['accuracy'], label='Train Accuracy')
        plt.plot(df['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy Evolution')
        plt.ylabel('Accuracy')
    else:
        plt.plot(df['mae'], label='Train MAE', color='orange')
        plt.plot(df['val_mae'], label='Val MAE', color='red')
        plt.title('Model Error (MAE) Evolution')
        plt.ylabel('Mean Absolute Error')
    
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'learning_curves_final.png')
    plt.savefig(save_path)
    print(f"Generated: {save_path}")
    plt.close()

def generate_metrics_evolution():
    data = {
        'Etapa': ['Etapa 4', 'Etapa 4', 'Etapa 5', 'Etapa 5', 'Etapa 6', 'Etapa 6'],
        'Metric': ['Accuracy', 'F1-Score', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score'],
        'Value': [0.20, 0.15, 0.64, 0.26, 0.72, 0.65]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Etapa', y='Value', hue='Metric', palette='viridis')
    
    plt.title('Evoluția Performanței (Etapa 4 -> 5 -> 6)')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.85, color='r', linestyle='--', label='Target Industrial Acc')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, 'metrics_evolution.png')
    plt.savefig(save_path)
    print(f"Generated: {save_path}")
    plt.close()

def discretize_action(left, right):
    threshold = 0.1
    if right - left > threshold: return "Stanga" # 0
    elif left - right > threshold: return "Dreapta" # 1
    else: return "Inainte" # 2

def generate_example_predictions():
    if not os.path.exists(CSV_FILENAME):
        print("CSV not found. Skipping examples.")
        return

    df = pd.read_csv(CSV_FILENAME)
    X = df[['senzor_stanga', 'senzor_centru', 'senzor_dreapta']].values
    y = df[['motor_stanga', 'motor_dreapta']].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = load_model(MODEL_FILENAME)

    indices = np.random.choice(len(X_test), 9, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Exemple Predicții (Test Set)', fontsize=16)

    for i, idx in enumerate(indices):
        ax = axes[i // 3, i % 3]
        
        sensors = X_test[idx]
        true_motor = y_test[idx]

        pred_motor = model.predict(sensors.reshape(1, 3), verbose=0)[0]
        
        true_label = discretize_action(true_motor[0], true_motor[1])
        pred_label = discretize_action(pred_motor[0], pred_motor[1])

        sensor_display = sensors.reshape(1, 3)
        
        ax.imshow(sensor_display, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['L', 'C', 'R'])
        ax.set_yticks([])

        color = 'green' if true_label == pred_label else 'red'
        
        title = f"True: {true_label}\nPred: {pred_label}"
        ax.set_title(title, color=color, fontweight='bold')

        text = f"S: {sensors.round(1)}\nM_True: {true_motor.round(2)}\nM_Pred: {pred_motor.round(2)}"
        ax.text(0.5, -0.5, text, ha='center', transform=ax.transAxes, fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'example_predictions.png')
    plt.savefig(save_path)
    print(f"Generated: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating visualizations...")
    generate_learning_curves()
    generate_metrics_evolution()
    generate_example_predictions()
    print("Done.")