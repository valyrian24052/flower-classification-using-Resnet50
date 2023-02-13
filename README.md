# flower-classification-using-Resnet50

This code implements a transfer learning approach for flower image classification using the ResNet 50 model pre-trained on the ImageNet dataset.

Dependencies 
PyTorch
TorchVision
Numpy
Scipy

This is a script for training a ResNet50 model on a Flowers dataset using PyTorch. The script starts by loading the image labels from a .mat file and creating a list of tuples with the image file path and label. It then creates a directory structure that mirrors the desired class structure, and copies the images to the corresponding class directories. The Flowers dataset is then loaded and transformed using the PyTorch's ImageFolder class and DataLoader utility. The pre-trained ResNet 50 model is loaded and fine-tuned by setting up the optimizer, loss function, and evaluation metric, and training the model for 10 epochs. The script ends by printing "Finished Training".
