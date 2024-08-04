import numpy as np
import tensorflow as tf
from model import build_model
from tensorflow.keras.models import load_model
import os

def load_preprocessed_data():
    """Load preprocessed data for prediction."""
    X_test = np.load('flower_classification/data/processed/X_test.npy')
    y_test = np.load('flower_classification/data/processed/y_test.npy')
    return X_test, y_test

def load_trained_model(model_path, input_shape, num_classes):
    """Load the trained model from file."""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        model = build_model(input_shape, num_classes)
        print(f"Initialized new model as no trained model found at {model_path}")
    return model

def make_predictions(model, X_test):
    """Make predictions using the trained model."""
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

def evaluate_predictions(predicted_classes, y_test):
    """Evaluate the predictions by comparing with the ground truth labels."""
    accuracy = np.mean(predicted_classes == y_test)
    return accuracy

if __name__ == "__main__":
    # Load preprocessed data
    X_test, y_test = load_preprocessed_data()
    
    # Define model parameters
    input_shape = X_test.shape[1:]
    num_classes = len(np.unique(y_test))
    
    # Load the trained model
    model_path = 'flower_classification/data/processed/trained_model.h5'
    model = load_trained_model(model_path, input_shape, num_classes)
    
    # Make predictions
    predicted_classes = make_predictions(model, X_test)
    
    # Evaluate predictions
    accuracy = evaluate_predictions(predicted_classes, y_test)
    print(f"Prediction accuracy: {accuracy:.2f}")
