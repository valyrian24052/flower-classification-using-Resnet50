import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    """Builds a CNN model using Keras."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, batch_size=32, epochs=10, validation_split=0.2):
    """Trains the model on the training data."""
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test data."""
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

if __name__ == "__main__":
    # Example usage
    input_shape = (128, 128, 3)  # Example input shape
    num_classes = 102  # Example number of classes (e.g., 102 flower categories)
    
    model = build_model(input_shape, num_classes)
    print(model.summary())
