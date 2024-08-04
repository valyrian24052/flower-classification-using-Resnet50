import subprocess

def main():
    # Step 1: Download the data
    subprocess.run(['python', 'flower_classification/src/download_data.py'], check=True)
    
    # Step 2: Preprocess the data
    subprocess.run(['python', 'flower_classification/src/data_preprocessing.py'], check=True)
    
    # Step 3: Train the model
    subprocess.run(['python', 'flower_classification/src/train.py'], check=True)
    
    # Step 4: Predict and evaluate (optional, you can run this step separately if needed)
    subprocess.run(['python', 'flower_classification/src/predict.py'], check=True)

if __name__ == "__main__":
    main()
