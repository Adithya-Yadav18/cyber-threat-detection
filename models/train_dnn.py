# models/train_dnn.py
# CyberShield AI - TensorFlow Deep Neural Network
# Trains a DNN classifier on CICIDS2017 dataset

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessor import run_preprocessing

def build_dnn_model(input_dim, num_classes):
    """Build the DNN architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_dnn():
    """Full DNN training pipeline."""
    print("=" * 60)
    print("  CyberShield AI - TensorFlow DNN Training")
    print("=" * 60)

    # Load preprocessed data
    X_train, X_test, y_train, y_test, features, le = run_preprocessing()

    num_classes = len(le.classes_)
    input_dim = X_train.shape[1]

    print(f"\n[*] Building DNN model...")
    print(f"    Input features : {input_dim}")
    print(f"    Output classes : {num_classes}")
    print(f"    Classes        : {list(le.classes_)}")

    model = build_dnn_model(input_dim, num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/saved_model/dnn_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    print("\n[*] Training DNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=2048,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[*] Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[+] Test Accuracy : {accuracy * 100:.2f}%")
    print(f"[+] Test Loss     : {loss:.4f}")

    # Save final model
    model.save('models/saved_model/dnn_model.h5')
    print("[+] DNN model saved to models/saved_model/dnn_model.h5")

    # Save training history
    with open('models/saved_model/dnn_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("[+] Training history saved.")
    print("\n[✓] DNN Training Complete!")

if __name__ == "__main__":
    train_dnn()