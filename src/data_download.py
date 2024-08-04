import os
import requests
import tarfile

def download_and_extract_tar(url, dest_folder, filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Download the file
    response = requests.get(url)
    tar_path = os.path.join(dest_folder, filename)
    with open(tar_path, "wb") as file:
        file.write(response.content)
    print(f"File {filename} downloaded successfully.")

    # Extract the tar file
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dest_folder)
    print(f"File {filename} extracted successfully.")
    
    # Remove the tar file
    os.remove(tar_path)
    print(f"File {filename} removed after extraction.")

if __name__ == "__main__":
    data_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    dest_folder = 'flower_classification/data/raw'
    filename = "102flowers.tgz"
    
    download_and_extract_tar(data_url, dest_folder, filename)
