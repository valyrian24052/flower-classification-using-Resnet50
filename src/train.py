import numpy as np
from model import build_model, train_model, evaluate_model

def main():
    """Main function to train and evaluate the model."""
    # Load preprocessed data
    X_train = np.load('flower_classification/data/processed/X_train.npy')
    X_test = np.load('flower_classification/data/processed/X_test.npy')
    y_train = np.load('flower_classification/data/processed/y_train.npy')
    y_test = np.load('flower_classification/data/processed/y_test.npy')
    
    # Build, train and evaluate the model
    model = build_model()
    model = train_model(model, X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    
    print(f"Model accuracy: {score:.2f}")

if __name__ == "__main__":
    main()
