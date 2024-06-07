
# MNIST Handwritten Digit Recognition with Grad-CAM

This project demonstrates an advanced implementation of a handwritten digit recognition system using the MNIST dataset and a Convolutional Neural Network (CNN). It includes data loading, preprocessing, data augmentation, model building, hyperparameter tuning, training, evaluation, and Grad-CAM visualization.

## Project Overview

- **Dataset**: MNIST
- **Model**: Convolutional Neural Network (CNN)
- **Hyperparameter Tuning**: Keras Tuner
- **Visualization**: Grad-CAM

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- TensorFlow
- Keras
- Matplotlib
- OpenCV
- NumPy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/debjit-mandal/mnist-gradcam-visualization.git
   cd mnist-gradcam-visualization
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:

   ```bash
   jupyter notebook MNIST.ipynb
   ```

## Project Structure

- `MNIST.ipynb`: The main Jupyter notebook containing the implementation.
- `requirements.txt`: List of required Python packages.

## Usage

1. Load and preprocess the MNIST dataset.
2. Define the CNN model and perform hyperparameter tuning using Keras Tuner.
3. Train the model with the best hyperparameters.
4. Evaluate the model on the test dataset.
5. Generate Grad-CAM heatmaps to visualize class-discriminative regions.

## Results

Include some results or sample images here.

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize where the model is looking when making predictions. This helps in understanding and interpreting the model's decisions.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun and can be downloaded from [here](http://yann.lecun.com/exdb/mnist/).
- The Grad-CAM implementation is based on the work by Ramprasaath R. Selvaraju et al.
