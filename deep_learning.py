"""
title: Neural Networks for Wages Prediction
description: Provides a deep learning approach based on a deep, feed-forward NN to predict wages.
"""

# Standard Library Imports
import os
import logging
import random
import numpy as np
import pandas as pd

# Third-Party Imports
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from dotenv import load_dotenv
from pyspark.sql.functions import col  # For data processing in Spark

# Custom Imports
from spark_session_management import create_spark_session
from ml_samples import get_ml_samples, get_training_metadata

# Load configuration from .env file
load_dotenv()


# ===============================
# Application Configuration
# ===============================
APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
TRAINING_DATA_FOLDER = os.path.join(APP_HOME, "training_data")

# ===============================
# Model Configuration (assumed to be in .env)
# ===============================
CONCEPTS = int(os.getenv("CONCEPTS", 6))  # Number of concepts (fixed)
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", 0.3))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 64))  # Embedding size from config

# ===============================
# Training Configuration
# ===============================
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))  # Number of training epochs
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))  # 20% for validation

# Set up Logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

#
# Function: Compile Wages Model
#
def compile_wages_model(vocab_size, EMBEDDING_SIZE=128, concepts=CONCEPTS):
    """
    Compiles the wages model with a two-layer neural network, with the second layer learning fixed concepts.

    Parameters:
        vocab_size (int): The size of the input features (e.g., vocabulary size).
        first_hidden_layer_size (int): The number of neurons in the first hidden layer (embedding size).
        concepts (int): The number of concepts to be learned, mapped to the second hidden layer's size.
                        Defaults to the value loaded from the .env file.

    Returns:
        model (tf.keras.Model): The compiled Keras model.
    """
    # Model definition
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.InputLayer(input_shape=(vocab_size,)),

        # First Dense Layer (Embedding layer or the first hidden layer)
        tf.keras.layers.Dense(first_hidden_layer_size, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(DROPOUT_RATE),  # Apply dropout

        # Second Dense Layer (Concepts layer) - Enforcing learning of 'concepts' (labor market indicators)
        tf.keras.layers.Dense(concepts, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(DROPOUT_RATE),  # Apply dropout

        # Output Layer (Predicting wage, 1 output node)
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    logger.debug("Model compiled with vocab_size={}, first_hidden_layer_size={}, concepts={}".format(vocab_size, first_hidden_layer_size, concepts))

    return model


# ===============================
# Training Logic and Helper Functions
# ===============================

def _rmse(y_true, y_pred, **kwargs):
    """
    Computes RMSE (Root Mean Squared Error) between true and predicted values.
    """
    return np.sqrt(mse(y_true, y_pred))

def _check_for_early_stopping(val_errors, threshold=0.05, min_epochs=30, window_size=10):
    """
    Check whether early stopping condition is met based on the last `window_size` validation errors.
    Early stopping is triggered if the validation error does not improve by at least `threshold` (5%) over the last `window_size` epochs.
    """
    if len(val_errors) < min_epochs:
        return False

    # Get the last `window_size` validation errors
    recent_errors = val_errors[-window_size:]

    # If the validation errors are less than `window_size`, we can't evaluate improvement over the window
    if len(recent_errors) < window_size:
        return False

    # Calculate the percentage change between the first and last validation error in the window
    initial_error = recent_errors[0]
    current_error = recent_errors[-1]

    # If the error hasn't improved by at least `threshold`, trigger early stopping
    improvement = (initial_error - current_error) / initial_error  # Positive improvement if current_error is smaller
    if improvement < threshold:
        return True

    return False


def _plot_errors(train_errors, val_errors, optimal_epoch=None, **kwargs):
    """
    Plot training and validation RMSE curves over epochs using Plotly, with an indication of the minimal validation error.
    """
    # Set default values for any additional parameters (optional)
    plot_title = kwargs.get('plot_title', "Training vs Validation RMSE")

    df = pd.DataFrame({
        "Epoch": np.arange(1, len(train_errors) + 1),
        "Train RMSE": train_errors,
        "Validation RMSE": val_errors
    })

    fig = px.line(df, x="Epoch", y=["Train RMSE", "Validation RMSE"], title=plot_title)
    fig.update_traces(mode='markers+lines')

    # Add gray shaded rectangle for early stopping if triggered
    if optimal_epoch:
        fig.add_vrect(
            x0=optimal_epoch, x1=len(train_errors),
            fillcolor="gray", opacity=0.2,
            line_width=0,
            annotation_text="Early Stopping Triggered",
            annotation_position="top right"
        )

    # Highlight the minimal validation error point
    min_val_error_idx = np.argmin(val_errors)
    min_val_error = val_errors[min_val_error_idx]
    fig.add_annotation(
        x=min_val_error_idx + 1,  # Epochs are 1-indexed
        y=min_val_error,
        text=f"Min Validation Error: {min_val_error:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-50
    )

    fig.show()


def compile_wages_model(VOCAB_SIZE):
    """
    Placeholder function for compiling the wages model.
    You should replace this with the actual model definition and compilation.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=VOCAB_SIZE, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_wages_model(train_X, train_y, val_X, val_y, **kwargs):
    """
    Trains the wages model based on provided data. If samples are not provided, they are fetched using Spark.
    Includes early stopping if the validation error increases.

    Parameters:
        train_X: Training features (must be provided).
        train_y: Training labels (must be provided).
        val_X: Validation features (must be provided).
        val_y: Validation labels (must be provided).
        **kwargs: Additional parameters for flexibility, such as `num_epochs`, `threshold`, `min_epochs`, etc.
    """
    # Pass kwargs to private methods and handle them there
    num_epochs = kwargs.get('num_epochs', int(os.getenv("NUM_EPOCHS", 100)))  # Default to 100 if the env variable is not found
    threshold = kwargs.get('threshold', 0.1)
    min_epochs = kwargs.get('min_epochs', 30)
    window_size = kwargs.get('window_size', 10)

    # Dynamically set VOCAB_SIZE based on the number of features in the training data
    VOCAB_SIZE = train_X.shape[1]  # Assuming train_X is a 2D array, VOCAB_SIZE is the number of features

    # Compile the model (assuming VOCAB_SIZE and other hyperparameters are defined)
    model = compile_wages_model(VOCAB_SIZE)

    # Initialize lists to track RMSE errors over epochs
    train_errors = []
    val_errors = []

    # Keep track of the optimal model (the one with the minimal validation error)
    optimal_model = None
    min_val_error = float('inf')  # Set the initial minimal validation error to infinity
    optimal_epoch = None  # The epoch where the minimal validation error occurred

    # Loop over epochs, using `num_epochs` as the number of iterations
    for epoch in range(num_epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs}")

        # Shuffle and batch the training data
        for batch_idx in range(0, len(train_X), BATCH_SIZE):
            # Select the batch
            X_batch = train_X[batch_idx: batch_idx + BATCH_SIZE]
            y_batch = train_y[batch_idx: batch_idx + BATCH_SIZE]

            # Train on the batch
            loss, mae = model.train_on_batch(X_batch, y_batch)

            # Recast loss and mae to numeric type (float)
            loss = float(loss)  # Ensure loss is scalar, not an array
            mae = float(mae)  # Recast MAE to scalar

        # Optionally, compute training error after each epoch
        train_error = _rmse(y_batch, model.predict(X_batch), **kwargs)  # Pass kwargs to _rmse
        train_errors.append(train_error)

        # Evaluate the model on validation data
        val_loss, val_mae = model.evaluate(val_X, val_y, verbose=0)
        val_error = _rmse(val_y, model.predict(val_X), **kwargs)  # Pass kwargs to _rmse
        val_errors.append(val_error)  # Track RMSE for validation as well

        print(f"[INFO] Epoch {epoch + 1}/{num_epochs} - Train RMSE: {train_error:.4f}, Validation RMSE: {val_error:.4f}")

        # Check if validation error is minimal (if so, update the optimal model)
        if val_error < min_val_error:
            min_val_error = val_error
            optimal_model = model  # Keep the current model as the optimal one
            optimal_epoch = epoch + 1  # Record the epoch where the optimal model was found

        # Check for early stopping
        if _check_for_early_stopping(val_errors, threshold=threshold, min_epochs=min_epochs, window_size=window_size):
            early_stop_epoch = epoch + 1  # Epochs are 1-indexed for reporting
            print(f"[INFO] Early stopping triggered at epoch {early_stop_epoch}.")
            break

    # Plot the error curves and indicate early stopping and the minimal validation error
    _plot_errors(train_errors, val_errors, optimal_epoch=optimal_epoch, **kwargs)  # Pass kwargs to _plot_errors

    return optimal_model  # Return the optimal model based on the minimal validation error
