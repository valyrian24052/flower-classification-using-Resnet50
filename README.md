# Flower Classification

This project aims to classify different types of flowers using a machine learning model. The model is built using a pretrained ResNet50 as the base model, followed by custom layers for the specific classification task.

## Directory Structure

## Installation

1. Clone the repository

   git clone https://github.com/yourusername/flower_classification.git
   cd flower_classification

2. Create and activate a virtual environment

   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the dependencies

   pip install -r requirements.txt

## Usage

### Running the Project

1. Download the data, preprocess it, train the model, and optionally make predictions and evaluate

   python main.py

   This will:
   - Download the dataset.
   - Preprocess the data and save it in the `data/processed` directory.
   - Train the model and save the trained model.
   - (Optionally) Make predictions and evaluate the model.

### Running Tests

To run tests for data preprocessing and the model:

   pytest flower_classification/tests

## Project Components

### `download_data.py`

Downloads and extracts the dataset from a given URL.

### `data_preprocessing.py`

Loads, preprocesses, and splits the dataset into training and testing sets. Preprocessed data is saved as numpy arrays in the `data/processed` directory.

### `model.py`

Defines and compiles a convolutional neural network (CNN) using a pretrained ResNet50 model as the base. Functions to train and evaluate the model are included.

### `train.py`

Loads preprocessed data, builds the model, trains the model, evaluates it, and saves the trained model.

### `predict.py`

Loads the preprocessed test data and the trained model, makes predictions, and evaluates the model's performance.

### `tests/test_data_preprocessing.py`

Contains tests for the data preprocessing functions.

### `tests/test_model.py`

Contains tests for the model building, training, and evaluation functions.

## Notes

- Ensure you have a stable internet connection to download the dataset and the pretrained ResNet50 model.
- Adjust the paths and parameters as needed for your specific use case.

## Acknowledgments

- The flower dataset is provided by the Visual Geometry Group at the University of Oxford.
- The ResNet50 model is pretrained on ImageNet and provided by the Keras applications module.