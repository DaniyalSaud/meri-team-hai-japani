# https://drive.google.com/file/d/1EK4z5JHpIeKuNaemF9Ux1fpq7AP5EDeI/view?usp=sharing

# This is the link for the ZIP file of the scripts and the model



# Offroad Segmentation Model - Step-by-Step Guide

This repository contains scripts for training and evaluating a semantic segmentation model for offroad scenes. The model is based on PyTorch Lightning and segmentation_models_pytorch (SMP).

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for training)
- Required Python packages:
  - torch
  - torchvision
  - pytorch-lightning
  - segmentation-models-pytorch
  - albumentations
  - opencv-python
  - matplotlib
  - numpy

Install all dependencies with:

```
pip install torch torchvision torchaudio pytorch-lightning segmentation-models-pytorch albumentations opencv-python matplotlib numpy
```

## Dataset Preparation

1. **Download the dataset**
   - Place the `Offroad_Segmentation_Training_Dataset` folder in the project root.
   - The folder should have the following structure:
     ```
     Offroad_Segmentation_Training_Dataset/
       train/
         Color_Images/
         Segmentation/
       val/
         Color_Images/
         Segmentation/
     ```
   - For testing, also place the `test_public_80` folder in the project root:
     ```
     test_public_80/
       Color_Images/
       Segmentation/
     ```

## Training the Model

Run the training script to train the segmentation model:

```
python train_script.py
```

- The script will train the model and save the following files in the project directory:
  - `offroad_segmentation_model_state_dict.pth` (model weights)
  - `offroad_segmentation_model.ckpt` (full Lightning checkpoint)
  - `offroad_segmentation_model.pkl` (pickled model, if enabled)

## Evaluating the Model

After training, evaluate the model on the test set:

```
python test_script.py
```

- The script will load the model from `offroad_segmentation_model.pkl` and print:
  - Mean IoU
  - Mean Dice
  - Mean Accuracy
  - Per-class metrics (IoU, Dice, Precision, Recall, Accuracy)

## Notes

- If you want to test a model saved as a state_dict or checkpoint, modify `test_script.py` to load the appropriate file.
- Ensure the dataset folder structure matches the expected format.
- For custom datasets or different architectures, update the dataset paths and model parameters in the scripts.

## Troubleshooting

- **ModuleNotFoundError**: Ensure all required packages are installed.
- **CUDA errors**: Make sure your GPU drivers and CUDA toolkit are properly installed.
- **FileNotFoundError**: Check that dataset and model files are in the correct locations.

## File Overview

- `train_script.py`: Trains the segmentation model.
- `test_script.py`: Evaluates the trained model on the test set.
- `Offroad_Segmentation_Training_Dataset/`: Training and validation data.
- `test_public_80/`: Test data.

---

For any issues, please check the script comments or contact the author.
