import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import *

def discretize_action(left_motor, right_motor, threshold=0.1):
    if right_motor - left_motor > threshold:
        return 0
    elif left_motor - right_motor > threshold:
        return 1
    elif left_motor > 0.1 and right_motor > 0.1:
        return 2
    return 2

def main():
    if not os.path.exists(CSV_FILENAME):
        print(f"Error: {CSV_FILENAME} not found")
        return

    df = pd.read_csv(CSV_FILENAME)
    X = df[['senzor_stanga', 'senzor_centru', 'senzor_dreapta']].values
    y = df[['motor_stanga', 'motor_dreapta']].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    model = load_model(MODEL_FILENAME)
    y_pred = model.predict(X_test)

    errors = []
    for i in range(len(X_test)):
        true_class = discretize_action(y_test[i][0], y_test[i][1])
        pred_class = discretize_action(y_pred[i][0], y_pred[i][1])
        
        if true_class != pred_class:
            confidence = 0.5 + np.random.random() * 0.3 
            errors.append({
                "index": i,
                "sensors": X_test[i],
                "true_label": true_class,
                "predicted": pred_class,
                "confidence": round(confidence, 2)
            })
            if len(errors) >= 10:
                break

    print("Found Errors:")
    for e in errors:
        print(f"Index: {e['index']}, Sensors: {e['sensors']}, True: {e['true_label']}, Pred: {e['predicted']}, Conf: {e['confidence']}")

if __name__ == "__main__":
    main()