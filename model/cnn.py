#cnn.py
"""
Defines a Keras-based 1D convolutional neural network (CNN) for binary classification on NSL-KDD data.
This model assumes the input data is in tabular format with features extracted from the preprocessed CSV files.
The architecture reshapes the feature vector (of dimension N) into a 1D channel sequence,
applies two convolutional layers with pooling, and then uses a dense layer for final prediction.
The final activation is a sigmoid for binary classification.

Author: [Your Name]
Date: [Date]
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_dim):
    """
    Creates and compiles a CNN model for binary classification.
    
    Args:
        input_dim (int): Number of input features (for example, from a CSV row without the label).
        
    Returns:
        model: A compiled tf.keras.Sequential model.
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),

        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

        
    # Compile the model with binary crossentropy loss and the Adam optimizer.
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # For a quick test, assume 128 features as produced by preprocess_cnn.py.
    dummy_input_dim = 128
    model = create_cnn_model(dummy_input_dim)
    model.summary()
