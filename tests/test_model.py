import pytest
import numpy as np
import tensorflow as tf
from src.model import build_model, train_model, evaluate_model

@pytest.fixture
def sample_data():
    # Create a sample dataset
    X_train = np.random.rand(10, 128, 128, 3)
    y_train = np.random.randint(0, 5, 10)
    X_test = np.random.rand(5, 128, 128, 3)
    y_test = np.random.randint(0, 5, 5)
    return X_train, y_train, X_test, y_test

def test_build_model():
    input_shape = (128, 128, 3)
    num_classes = 5
    model = build_model(input_shape, num_classes)
    
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_shape == (None, num_classes)

def test_train_model(sample_data):
    X_train, y_train, X_test, y_test = sample_data
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    model = build_model(input_shape, num_classes)
    
    history = train_model(model, X_train, y_train, epochs=1)
    
    assert 'accuracy' in history.history
    assert 'val_accuracy' in history.history

def test_evaluate_model(sample_data):
    X_train, y_train, X_test, y_test = sample_data
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    model = build_model(input_shape, num_classes)
    
    train_model(model, X_train, y_train, epochs=1)
    accuracy = evaluate_model(model, X_test, y_test)
    
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
