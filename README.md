# Transfer Learning with ResNet50 on MNIST

This project implements transfer learning using the ResNet50 architecture on the MNIST dataset. The goal is to leverage pre-trained models to improve the performance of digit classification tasks with minimal training.

## Project Overview

- **Transfer Learning**: The project applies transfer learning by using a pre-trained ResNet50 model and fine-tuning it on the MNIST dataset.
- **Spatial Transformer Network**: The project includes a spatial transformer module to enhance feature extraction.
- **MNIST Dataset**: The MNIST dataset is used for training and evaluation.

## Project Structure

- **TransferLearning.py**: The main Python script that implements the transfer learning workflow using ResNet50.
- **SpatialTransformer.py**: Contains the code for the spatial transformer module used for augmenting the feature extraction process.
- **data/MNIST**: Contains the MNIST dataset in both raw and processed formats.

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python 3.x is installed on your machine.
- **Required Libraries**: Install the required libraries listed in `requirements.txt`.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/TransferLearning-Resnet50-MNIST.git
    cd TransferLearning-Resnet50-MNIST
    ```

2. **Install Dependencies**:
    Install the necessary dependencies for running the project:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Run the Transfer Learning Script**:
    Execute the transfer learning script to train the ResNet50 model on the MNIST dataset:
    ```bash
    python TransferLearning.py
    ```

2. **Evaluate the Model**:
    The script will output the results of the training and evaluation on the MNIST test set.

## Project Workflow

1. **Data Loading**: Load the MNIST dataset from the `data/MNIST/` directory.
2. **Transfer Learning**: Use ResNet50 pre-trained weights to fine-tune the model on the MNIST dataset.
3. **Spatial Transformer**: Augment the feature extraction process using the spatial transformer module.
4. **Evaluation**: Evaluate the model on the MNIST test set to check the performance after fine-tuning.
