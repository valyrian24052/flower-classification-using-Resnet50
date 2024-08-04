import pytest
import numpy as np
import os
from src.data_preprocessing import load_images_from_folder, preprocess_data, split_data

@pytest.fixture
def sample_data():
    # Create a sample dataset
    images = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    labels = np.random.randint(0, 5, 10, dtype=np.uint8)
    return images, labels

def test_load_images_from_folder(tmp_path):
    # Create a temporary directory with some images
    class_dirs = ['class1', 'class2']
    for class_dir in class_dirs:
        os.makedirs(tmp_path / class_dir)
        for i in range(5):
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img)
            img.save(tmp_path / class_dir / f'{i}.jpg')
    
    images, labels = load_images_from_folder(tmp_path)
    assert len(images) == 10
    assert len(labels) == 10

def test_preprocess_data(sample_data):
    images, labels = sample_data
    images, labels = preprocess_data(images, labels)
    
    # Check if images are normalized
    assert images.min() >= 0.0
    assert images.max() <= 1.0
    
    # Check if labels are unchanged
    assert np.array_equal(labels, sample_data[1])

def test_split_data(sample_data):
    images, labels = sample_data
    X_train, X_test, y_train, y_test = split_data(images, labels, test_size=0.2)
    
    # Check the split sizes
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
