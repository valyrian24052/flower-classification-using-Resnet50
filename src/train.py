import numpy as np
from model import build_model, train_model, evaluate_model
import os

def main():
    """Main function to train and evaluate the model."""
    # Load preprocessed data
    X_train = np.load('flower_classification/data/processed/X_train.npy')
    X_test = np.load('flower_classification/data/processed/X_test.npy')
    y_train = np.load('flower_classification/data/processed/y_train.npy')
    y_test = np.load('flower_classification/data/processed/y_test.npy')
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    # Build, train and evaluate the model
    model = build_model(input_shape, num_classes)
    train_model(model, X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    model_path = 'flower_classification/data/processed/trained_model.h5'
    model.save(model_path)
    print(f"Model saved at {model_path}")
    
    print(f"Model accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
