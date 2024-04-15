# Multilayer Perceptron for Binary Classification

## Project Overview

This project implements a multilayer perceptron (MLP), a type of feedforward artificial neural network, from scratch using C++. The MLP is designed to classify data into two categories, making it ideal for binary classification tasks. This particular implementation focuses on the classification of breast mass cells as either malignant or benign based on cell nucleus characteristics extracted via fine-needle aspiration.

## Key Features

- **Multilayer Architecture**: Includes at least two hidden layers in the network, customizable through configuration files or command-line arguments.
- **Custom Activation Functions**: Utilizes sigmoid and softmax activation functions for different layers.
- **Backpropagation and Gradient Descent**: Employs these methods for training the network on a dataset.
- **Data Manipulation**: Includes a program to split the dataset into training and validation sets.
- **Performance Metrics**: Evaluates the model using loss and accuracy metrics, and displays learning curves during training.
- **No External Libraries for Neural Network Mechanics**: All neural network functionalities are implemented from scratch, though libraries are used for linear algebra and data visualization.

## Installation and Setup

Ensure you have Python installed on your system to set up the environment and dependencies.

1. Clone the repository:
```bash
git clone https://github.com/achatela/Multilayer-perceptron.git
```

2. Navigate to the project directory:
```bash
cd Multilayer-perceptron
```

3. Run the setup script to create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate # To exit the virtual environment, run 'deactivate'
pip install -r requirements.txt
```

## Compilation and Execution

This project uses a Makefile for compiling the C++ code. To prepare the data and train the model, you can simply run the following command while in the virtual environnement:

```bash
make shuffle
```

If you want to play with the hyperparameters you can run the following command:

```bash
make
python3 utils/separate_dataset.py data.csv
./train <data_training.csv> <data_validation.csv> <epochs> <learning_rate> "<hidden_layers>" ("8 4" for example)
```
