# 🔬 Cell Segmentation UNet Model 

This project provides a robust implementation of a **UNet** convolutional neural network for the task of instance-level cell nucleus segmentation. The model is trained on data from the Kaggle 2018 Data Science Bowl and is contained within the Jupyter Notebook, **`Cell_segmentation_CNN.ipynb`**.

The workflow uses **PyTorch** and the specialized **`segmentation-models-pytorch`** library.

-----


## 🚀 Model Architecture & Performance

### Architecture

| Component | Detail | Purpose |
| :--- | :--- | :--- |
| **Model** | **UNet** (Encoder-Decoder) | Standard architecture for pixel-wise semantic segmentation. |
| **Encoder** | **ResNet34** | Used as the backbone for robust feature extraction. |
| **Encoder Weights** | `imagenet` | Utilizes **Transfer Learning** to leverage features learned from a large dataset, improving convergence and generalization. |
| **Loss Function** | **`DiceBCELoss`** (Custom) | A combination of Dice Loss and Binary Cross-Entropy, critical for handling the severe class imbalance in segmentation tasks. |


### Key Performance Metrics (on Validation Set)

The model was trained for 5 epochs and achieved strong results:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Validation IoU (Per Image Avg)** | $\approx 0.7028$ | High overlap between predicted and true masks. |
| **Validation Dice Score (Per Image Avg)** | $\approx 0.7973$ | High similarity between predicted and true masks. |
| **Per-Pixel F1 Score (Weighted)** | $\approx 0.9633$ | High pixel-level accuracy in classifying foreground/background. |
| **ROC AUC** | $\approx 0.9997$ | Excellent model separability across all classification thresholds. |

-----


## 🛠️ Setup and Dependencies

This project requires a Python environment with PyTorch and standard data science libraries.

### Requirements

To run the notebook, install the following packages:

```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch
pip install numpy matplotlib scikit-learn
pip install albumentations opencv-python tqdm
```

## 📥 Dataset Link

The data used for training and evaluating this model is the **Kaggle 2018 Data Science Bowl** dataset.

| Dataset Name | Source Link |
| :--- | :--- |
| **2018 Data Science Bowl** | [https://www.kaggle.com/c/dsbowl-2018](https://www.kaggle.com/c/dsbowl-2018) |


## ⚙️ Workflow and Execution

The workflow in `Cell_segmentation_CNN.ipynb` is structured as follows:

1.  **Setup & Imports:** Install dependencies and define the CUDA/CPU device.
2.  **Data Loading:** Mount Google Drive and unzip the `stage1_train.zip` and `stage1_test.zip` files.
3.  **Model & Loss Definition:** Define the `CellDataset` and the custom `DiceBCELoss`.
4.  **Training:** Instantiate the UNet model, split the data into train/validation sets, and run the 5-epoch training loop.
5.  **Evaluation:** Calculate and display comprehensive metrics: Dice/IoU scores, Classification Report, Confusion Matrix, and the ROC Curve.
6.  **Inference:** Save the final model weights and provide an inference function (`segment_cell_image`) to test new images.

-----


## 🖼️ Visual Inference 

These examples demonstrate the UNet model's ability to accurately identify and segment individual cell nuclei in unseen images from the test set.

<img width="318" height="490" alt="image" src="https://github.com/user-attachments/assets/7e184f98-a963-4e17-9ffb-9673b9703c29" /> 


## 💾 Model Saving and Loading

The final trained model weights are saved using PyTorch's state dictionary format.


### Saving (Used in the Notebook)

The command used to save the model is:

```python
torch.save(trained_model.state_dict(), "unet_final_model.pth")
```


### Loading the Model for Inference

To use the saved model, you must first define the exact UNet architecture and then load the saved weights:

```python
import torch
import segmentation_models_pytorch as smp

# 1. Define the model architecture exactly as it was trained
loaded_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None, # Use None, as we are loading learned weights
    in_channels=3,
    classes=1,
).to(device)

# 2. Load the state dictionary
loaded_model.load_state_dict(torch.load("unet_final_model.pth"))
loaded_model.eval() # Set to evaluation mode for inference
```
