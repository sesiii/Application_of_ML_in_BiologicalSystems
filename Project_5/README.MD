# Application of Machine Learning in Biological Systems - Project 5

## Overview

This project involves the application of a Convolutional Neural Network (CNN) to classify images of fine needle aspiration (FNA) samples into benign and malignant categories. The dataset used for this project is located in the `Project_5/Dataset2/Dataset2` directory.

## Directory Structure

```
/c:/Users/sasan/Application_of_ML_in_BiologicalSystems/Project_5/
│
├── Dataset2/
│   ├── FNA/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│
├── README.md
└── script.py
```

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pillow
- scikit-learn
- Matplotlib

## Installation

To install the required packages, run:

```bash
pip install tensorflow numpy pillow scikit-learn matplotlib
```

## Usage

1. **Set the data directories**: Ensure the dataset is placed in the correct directory structure as shown above.

2. **Run the script**: Execute the script to train the CNN model and evaluate its performance.

```bash
python a.py
```

## Script Details

- **Data Loading and Preprocessing**: The script loads images from the benign and malignant directories, resizes them to 64x64 pixels, and normalizes the pixel values.

- **Model Building**: A CNN model is built using TensorFlow and Keras, consisting of convolutional layers, max-pooling layers, and dense layers.

- **Model Training**: The model is trained on the training dataset and validated on the validation dataset. Training history is saved to `output.txt`.

- **Evaluation**: The model's performance is evaluated on the training and validation datasets. Predictions on the test dataset are saved to `test_predictions.txt`.

- **Visualization**: Training and validation loss and accuracy are plotted and saved as `training_validation_loss_accuracy.png`.

## Results

The training and validation loss and accuracy for each epoch are saved in `output.txt`. The final predictions on the test dataset are saved in `test_predictions.txt`. The training and validation loss and accuracy plots are saved as `training_validation_loss_accuracy.png`.

## GPU Usage

The script checks for GPU availability and uses it for training if available. Otherwise, it falls back to using the CPU.

## License

This project is licensed under the MIT License.

## Acknowledgements

- TensorFlow and Keras for providing the deep learning framework.
- scikit-learn for data splitting utilities.
- Matplotlib for plotting utilities.
- Pillow for image processing utilities.















































































