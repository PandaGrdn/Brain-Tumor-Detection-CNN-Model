This project implements a Convolutional Neural Network (CNN) to classify MRI images as either containing a brain tumor (yes) or not (no).

Features
CNN Architecture: Two convolutional layers, max pooling, dropout, and fully connected layers.
K-Fold Cross-Validation: Ensures robust evaluation.

Preprocessing: Images are resized to  224 × 224 normalized.
Dataset
Organize your dataset as follows:
root_dir/
    yes/  # Images with brain tumors
    no/   # Images without brain tumors
Adjust Parameters: Modify num_epochs, batch_size, learning_rate, and num_folds as needed in the script.
Results
The model outputs fold-wise and mean accuracy:
  Fold 1: Accuracy = 0.85
  Fold 2: Accuracy = 0.87
  ...
  Mean Accuracy: 0.86
Model Architecture:
  Two Conv2D layers with ReLU and MaxPooling.
  Flatten → Fully Connected Layer → Dropout → Output Layer.
  Input size: 224×224×3.
