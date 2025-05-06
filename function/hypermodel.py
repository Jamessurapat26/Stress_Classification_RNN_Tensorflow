import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend 
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import json
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

# Define n_steps as a global variable
n_steps = 5
class MyHyperModel(kt.HyperModel):

    def __init__(self):
        self.trial_id = 1
        print("MyHyperModel initialized.")
        print(f'trial_id: {self.trial_id}')

    def build(self, hp):
        # Get parameters from hp
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.2, step=0.1)
        num_layers = hp.Int('num_layers', min_value=1, max_value=5)
        optimizer_name = hp.Choice('optimizer', ['adam', 'sgd'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        timesteps = n_steps # Use the defined n_steps

        # Define input shape
        features = 17
        input_shape = (timesteps, features)

        model = Sequential()

        # First layer
        model.add(SimpleRNN(units, activation='tanh', input_shape=input_shape,
                            return_sequences=(num_layers > 1)))
        if dropout > 0:
            model.add(Dropout(dropout))

        # Middle layers (if any)
        for i in range(num_layers - 2):
            model.add(SimpleRNN(units, activation='tanh', return_sequences=True))
            if dropout > 0:
                model.add(Dropout(dropout))

        # Last RNN layer
        if num_layers > 1:
            model.add(SimpleRNN(units, activation='tanh', return_sequences=False))
            if dropout > 0:
                model.add(Dropout(dropout))

        # Output layer for 4 classes
        model.add(Dense(4, activation='softmax'))

        # Choose optimizer
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

        # Compile model
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        return model

    def fit(self, hp, model, X, y, X_test, y_test, *args, **kwargs):
        # Ensure input is numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        units = hp.get('units')
        dropout = hp.get('dropout')
        num_layers = hp.get('num_layers')
        optimizer_name = hp.get('optimizer')
        learning_rate = hp.get('learning_rate')

        n_splits = 5
        trial = self.trial_id
        print(f"Trial {trial} - Starting training...")

        batch_size = hp.Int('batch_size', 8, 64, step=8)
        epochs = hp.Int('epochs', 10, 100)

        kf = KFold(n_splits=n_splits)

        val_scores = []
        fold = 0

        for train_idx, val_idx in kf.split(X): # Use the sequenced data X_seq here
            print("Training fold:", fold + 1)
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            print(f"Fold {fold+1} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"Fold {fold+1} - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            # print(y_train)

            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[
                    early_stopping,
                ]
            )

            val_scores.append(np.max(history.history['val_accuracy']))

            fold += 1

        os.makedirs(f'result/{trial}', exist_ok=True)

        mean_val_accuracy = np.mean(val_scores)
        y_pred = model.predict(X_test)
        print("Predictions shape:", y_pred.shape)

        accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
        print("Test accuracy:", accuracy)

        # Display classification report with proper class labels
        print("\nClassification Report:")
        target_names = ['High', 'Low', 'Medium', 'Normal']
        report = classification_report(y_test, np.argmax(y_pred, axis=1), 
                    target_names=target_names)  
        print(report)

        # Fix the F1 score calculation by using argmax
        f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='weighted')
        print("F1 Score:", f1)
        # Save confusion matrix as image
        cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'result/{trial}/confusion_matrix_{trial}.png')
        plt.close()
        print(f"Confusion matrix saved as confusion_matrix_{trial}.png")


        metrics_data = {
        'trial': trial,
        'units': units,
        'dropout': dropout,
        'num_layers': num_layers,
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'accuracy': float(accuracy),  # Convert numpy types to Python native types
        'mean_val_accuracy': mean_val_accuracy,
        'f1_score': f1,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        metrics_filename = f'result/{trial}/metrics_data_{trial}.json'

        # Save metrics data to JSON file
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(metrics_data)

        print(f"Trial {trial} completed. Metrics saved to {metrics_filename}.")
        self.trial_id = trial + 1

        backend.clear_session()  # Clear the session to free up resources

        return mean_val_accuracy