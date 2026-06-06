# models/train_lstm.py
# CyberShield AI - LSTM Model
# Sequence-based attack detection using LSTM

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessor import run_preprocessing

def build_lstm_model(input_dim, num_classes):
    """Build LSTM architecture."""
    model = tf.keras.Sequential([
        # Reshape flat features into sequences of 6 timesteps x 13 features
        tf.keras.layers.Reshape((6, 13), input_shape=(input_dim,)),

        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(64, return_sequences=False),
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

def train_lstm():
    """Full LSTM training pipeline."""
    print("=" * 60)
    print("  CyberShield AI - LSTM Training")
    print("=" * 60)

    # Load preprocessed data
    X_train, X_test, y_train, y_test, features, le = run_preprocessing()

    # LSTM needs input_dim divisible by timesteps
    # We use 78 features but trim to 78 (6x13=78) - perfect fit
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    # Trim to 78 features (already 78, just confirming)
    X_train = X_train[:, :78]
    X_test = X_test[:, :78]

    print(f"\n[*] Building LSTM model...")
    print(f"    Input shape : (6 timesteps x 13 features)")
    print(f"    Classes     : {list(le.classes_)}")

    model = build_lstm_model(78, num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/saved_model/lstm_model.h5',
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

    print("\n[*] Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=2048,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[*] Evaluating LSTM model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[+] Test Accuracy : {accuracy * 100:.2f}%")
    print(f"[+] Test Loss     : {loss:.4f}")

    # Save model
    model.save('models/saved_model/lstm_model.h5')
    print("[+] LSTM model saved to models/saved_model/lstm_model.h5")

    # Save history
    with open('models/saved_model/lstm_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("[+] Training history saved.")
    print("\n[✓] LSTM Training Complete!")

if __name__ == "__main__":
    train_lstm()