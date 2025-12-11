import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
from config.config import CSV_FILENAME, MODEL_FILENAME, TEST_SIZE

if not os.path.exists(CSV_FILENAME):
    print(f"EROARE: Fișierul {CSV_FILENAME} nu există.")
    print("Rulați întâi 'simulator.py' în modul 'MANUAL_LOGGING' pentru a genera date.")
else:
    print(f"Se încarcă datele din {CSV_FILENAME}...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset = pd.read_csv(CSV_FILENAME)
    
    dataset = dataset[(dataset['motor_stanga'] != 0) | (dataset['motor_dreapta'] != 0)]
    
    print(f"S-au încărcat {len(dataset)} eșantioane de date.")

    X = dataset[['senzor_stanga', 'senzor_centru', 'senzor_dreapta']].values
    Y = dataset[['motor_stanga', 'motor_dreapta']].values
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
    
    print(f"Set antrenare: {len(X_train)} eșantioane")
    print(f"Set validare: {len(X_val)} eșantioane")

    print("Se definește modelul...")
    
    model = Sequential()
    model.add(Dense(16, input_dim=3, activation='relu'))
    
    model.add(Dense(8, activation='relu'))
    
    model.add(Dense(2, activation='tanh'))
    
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[
                      keras.metrics.BinaryAccuracy(),
                      keras.metrics.FalseNegatives(),
                      'accuracy',
                      'mse'
                  ],
                  )
    
    model.summary()

    print("Se antrenează modelul...")
    
    # history = model.fit(
    #     X_train, Y_train,
    #     validation_data=(X_val, Y_val),
    #     # epochs=50,
    #     # batch_size=32,
    #     epochs=50,
    #     batch_size=32,
    #     verbose=1
    # )

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        verbose=2,
        validation_batch_size=None,
        callbacks=None,
    )
    
    print(f"Se salvează modelul antrenat ca {MODEL_FILENAME}...")
    model.save(MODEL_FILENAME)
    print("Salvare completă.")

    print("Se generează graficul de antrenare...")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Eroare antrenare (loss)')
    plt.plot(history.history['val_loss'], label='Eroare validare (val_loss)')
    plt.title('Istoricul antrenării modelului')
    plt.xlabel('Epoci')
    plt.ylabel('Eroare (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph.png')
    plt.show()
    print("Grafic salvat ca 'graph.png'.")