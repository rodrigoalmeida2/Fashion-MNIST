# Fashion MNIST Classifier

This project implements a deep learning model using TensorFlow and Keras to classify images from the **Fashion MNIST dataset**. The code is structured using **Object-Oriented Programming (OOP)** for better modularity and maintainability.

## 📌 Features
- Loads and preprocesses the **Fashion MNIST dataset**.
- Builds a **Neural Network model** with Keras.
- Trains the model with an adjustable number of epochs.
- Evaluates the model on test data.
- Saves the trained model.
- Generates accuracy plots.
- Predicts the category of test images.

## 🛠 Requirements
Make sure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

You can install dependencies using:
```bash
pip install tensorflow matplotlib numpy
```

## 🚀 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-mnist-classifier.git
   cd fashion-mnist-classifier
   ```
2. Run the script:
   ```bash
   python fashion_mnist.py
   ```
3. The script will:
   - Train the model on Fashion MNIST dataset.
   - Save the model as `modelo.h5`.
   - Plot the accuracy graph.
   - Evaluate model performance.
   - Predict the label for a test image.

## 📊 Model Architecture
The neural network consists of:
- **Flatten Layer** (Input: 28×28 images → 1D vector)
- **Dense Layer (256 neurons, ReLU activation)**
- **Dropout Layer (20%)**
- **Dense Layer (10 neurons, Softmax activation)**

## 📜 License
This project is licensed under the MIT License.

