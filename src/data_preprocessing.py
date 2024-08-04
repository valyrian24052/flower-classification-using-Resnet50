import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, image_size=(128, 128)):
    """Load images from a folder, resize them, and return as numpy arrays."""
    images = []
    labels = []
    for class_idx, class_folder in enumerate(os.listdir(folder)):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = Image.open(img_path)
                img = img.resize(image_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_idx)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    """Normalize images and return images and labels."""
    images = images / 255.0  # Normalize pixel values to [0, 1]
    return images, labels

def split_data(images, labels, test_size=0.2):
    """Split images and labels into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_folder = 'flower_classification/data/raw/102flowers'
    images, labels = load_images_from_folder(data_folder)
    images, labels = preprocess_data(images, labels)
    X_train, X_test, y_train, y_test = split_data(images, labels)
    
    # Save preprocessed data (optional, based on your requirement)
    np.save('flower_classification/data/processed/X_train.npy', X_train)
    np.save('flower_classification/data/processed/X_test.npy', X_test)
    np.save('flower_classification/data/processed/y_train.npy', y_train)
    np.save('flower_classification/data/processed/y_test.npy', y_test)
    
    print("Data preprocessing completed successfully.")
