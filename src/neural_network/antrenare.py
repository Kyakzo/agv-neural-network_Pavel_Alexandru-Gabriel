import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as pd_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config import *

def discretize_action(left_motor, right_motor, threshold=0.2):
    if right_motor - left_motor > threshold:
        return 1 # Viraj Stanga (Motor Drept > Motor Stang)
    elif left_motor - right_motor > threshold:
        return 2 # Viraj Dreapta
    elif left_motor > 0 and right_motor > 0:
        return 0 # Inainte
    return 3 # Stationare sau eroare

def main():
    print("=== Începere antrenare model ===")

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs'))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Creat directorul de rezultate: {results_dir}")

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Creat directorul docs: {docs_dir}")

    if not os.path.exists(CSV_FILENAME):
        print(f"Eroare: Fișierul de date {CSV_FILENAME} nu există!")
        return

    df = pd.read_csv(CSV_FILENAME)

    if len(df) < 50:
        print("Avertisment: Prea puține date pentru antrenare.")
    
    X = df[['senzor_stanga', 'senzor_centru', 'senzor_dreapta']].values
    y = df[['motor_stanga', 'motor_dreapta']].values

    print(f"Date încărcate: {len(df)} mostre.")

    # 2. Împărțire date (Train / Validation / Test)
    # 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
    # Din Temp: 50% Val, 50% Test => 15% Val, 15% Test din total
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Arhitectura aleasă este una de tip "Feed-Forward" (Sequential)
    # Strat Input: 3 neuroni (corespunzător celor 3 senzori)
    # Strat Ascuns 1: 16 neuroni, activare ReLU pentru non-linearitate
    # Strat Ascuns 2 (Adăugat în Etapa 6): 8 neuroni
    # Strat Output: 2 neuroni (Motor Stânga, Motor Dreapta)
    # Activare Lineară deoarece facem REGRESIE (valori continue [-1, 1])
    model = Sequential([
        Dense(16, input_shape=(3,), activation='relu'),
        Dense(8, activation='relu'),
        Dense(2, activation='linear')
    ])

    # model = Sequential([
    #     Dense(16, input_shape=(3,), activation='relu'),
    #     Dense(8, activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(2, activation='linear')])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Configurare Callbacks
    # EarlyStopping: Oprește antrenarea dacă val_loss nu scade timp de 5 epoci
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    # ReduceLROnPlateau: Scade Learning Rate-ul dacă performanța stagnează
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    print(f"\nAntrenare cu: Epochs={EPOCHS}, Batch_size={BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping, lr_scheduler]
    )

    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(results_dir, 'training_history.csv')
    history_df.to_csv(history_path, index_label='epoch')
    print(f"Istoric antrenare salvat în: {history_path}")

    print("\n=== Evaluare pe Test set ===")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")

    # Generare predicții pentru calculul metricilor discrete
    y_pred = model.predict(X_test)

    y_test_classes = [discretize_action(l, r) for l, r in y_test]
    y_pred_classes = [discretize_action(l, r) for l, r in y_pred]
    
    acc = accuracy_score(y_test_classes, y_pred_classes)
    f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
    
    print(f"\nMetrici calculate (prin discretizare):")
    print(f"Acuratețe: {acc*100:.2f}% (Target: >= 65%)")
    print(f"F1-Score (macro): {f1:.4f} (Target: >= 0.60)")

    metrics = {
        "test_loss_mse": float(test_loss),
        "test_mae": float(test_mae),
        "accuracy": float(acc),
        "f1_score_macro": float(f1)
    }
    metrics_path = os.path.join(results_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrici test salvate în: {metrics_path}")

    hyperparams = {
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "TEST_SIZE": 0.30,
        "VALIDATION_SPLIT": 0.50, # Applied to temp
        "OPTIMIZER": "adam",
        "LOSS_FUNCTION": "mse",
        "LAYERS": [16, 8, 2]
    }
    
    yaml_content = ""
    for key, value in hyperparams.items():
        yaml_content += f"{key}: {value}\n"
        
    yaml_path = os.path.join(results_dir, 'hyperparameters.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Hiperparametri salvați în: {yaml_path}")

    save_path = MODEL_FILENAME
    
    model.save(save_path)
    print(f"\nModel salvat în: {save_path}")

    print("\n" + "="*50)
    print("TABEL JUSTIFICARE HIPERPARAMETRI")
    print("="*50)
    print(f"{ 'Parametru':<25} | { 'Valoare':<20}")
    print("-" * 47)
    print(f"{ 'Nr. Epoci':<25} | {EPOCHS:<20}")
    print(f"{ 'Batch Size':<25} | {BATCH_SIZE:<20}")
    print(f"{ 'Optimizer':<25} | {'Adam':<20}")
    print(f"{ 'Loss Function':<25} | {'MSE (Regression)':<20}")
    print(f"{ 'Train Split':<25} | {'70%':<20}")
    print(f"{ 'Validation Split':<25} | {'15%':<20}")
    print(f"{ 'Test Split':<25} | {'15%':<20}")
    print("="*50)

    print("Se generează graficul de antrenare...")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Eroare antrenare (loss)')
    plt.plot(history.history['val_loss'], label='Eroare validare (val_loss)')
    plt.title('Istoricul antrenării modelului')
    plt.xlabel('Epoci')
    plt.ylabel('Eroare (MSE)')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(docs_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    # plt.show()
    print(f"Grafic salvat în '{loss_curve_path}'.")

if __name__ == "__main__":
    main()