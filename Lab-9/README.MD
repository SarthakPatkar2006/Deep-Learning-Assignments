# 🖼️ Image Classification: Vision Transformer (ViT) vs CNN

<p align="center">
  <b>Deep Learning | Computer Vision | Transfer Learning | Model Comparison</b>
</p>

<p align="center">
  A comparative study of Vision Transformer (ViT) and Convolutional Neural Network (CNN) architectures for multi-class image classification using the CIFAR-10 dataset.
</p>

---

## 📌 Project Overview

This project implements and compares two popular deep learning architectures for image classification:

1. **Vision Transformer (ViT)** – A Transformer-based architecture that uses self-attention mechanisms to understand relationships between image patches.
2. **Convolutional Neural Network (CNN)** – A convolution-based architecture that extracts spatial and hierarchical features from images.

The project uses the CIFAR-10 dataset, which contains images belonging to 10 different object categories.

Both models use pre-trained weights and Transfer Learning. Their feature extraction layers are frozen, while their classification heads are trained on the CIFAR-10 dataset.

The objective is to evaluate both architectures based on classification performance, inference time, and their ability to classify different image categories.

## 🎯 Project Objectives

- Understand the working principles of CNN and Vision Transformer architectures.
- Implement transfer learning using pre-trained models.
- Perform image preprocessing and normalization.
- Fine-tune classification heads for a 10-class classification problem.
- Compare ViT and ResNet-18 performance.
- Evaluate models using Accuracy, Precision, Recall, and F1-score.
- Analyze classification errors using confusion matrices.
- Compare inference time per image.
- Visualize predictions from both models.

## ✨ Key Features

- **Dual Architecture Implementation:** Implements both ViT and ResNet-18.
- **Transfer Learning:** Uses pre-trained ImageNet weights.
- **Frozen Feature Extractors:** Retains pre-trained feature representations.
- **Custom Classification Heads:** Replaces the final classification layers for 10 CIFAR-10 classes.
- **Performance Comparison:** Evaluates both architectures using common metrics.
- **Confusion Matrix Visualization:** Identifies correctly and incorrectly classified categories.
- **Accuracy Comparison Graph:** Provides a visual comparison of model accuracy.
- **Image Prediction:** Displays actual labels alongside predictions from both models.
- **GPU Support:** Automatically detects CUDA availability.

## 📂 Dataset

The project uses the **CIFAR-10 Image Classification Dataset** provided by Torchvision.

Dataset: [CIFAR-10 – University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

CIFAR-10 consists of 60,000 colour images divided into 10 different classes.

### Dataset Classes

| Class ID | Category |
|---|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

### Dataset Configuration

For faster experimentation and execution in Google Colab, the notebook uses subsets of the original dataset.

| Parameter | Value |
|---|---:|
| Original Training Images | 50,000 |
| Original Testing Images | 10,000 |
| Selected Training Images | 10,000 |
| Selected Testing Images | 2,000 |
| Number of Classes | 10 |
| Image Channels | 3 (RGB) |
| Original Image Size | 32 × 32 |
| Model Input Size | 224 × 224 |
| Training Batch Size | 32 |

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Torchvision | Dataset loading and image transformations |
| Hugging Face Transformers | Vision Transformer implementation |
| ResNet-18 | CNN architecture |
| NumPy | Numerical computations |
| Matplotlib | Image and graph visualization |
| Seaborn | Confusion matrix and performance visualization |
| Scikit-learn | Classification evaluation metrics |
| Google Colab | Notebook execution environment |

## 🏗️ Model Architecture

The project compares two different approaches to extracting image features and performing classification.

### 1. Vision Transformer (ViT)

<https://huggingface.co/google/vit-base-patch16-224>

The project uses the pre-trained `google/vit-base-patch16-224` model.

Vision Transformer divides an input image into smaller patches and processes them using Transformer encoder layers.

**Architecture Workflow:**

```text
          Input Image
          (224 × 224)
               |
               ▼
         Image Patching
          (16 × 16)
               |
               ▼
        Patch Embeddings
               |
               ▼
       Positional Embeddings
               |
               ▼
       Transformer Encoder
        (Self-Attention)
               |
               ▼
       Contextual Features
               |
               ▼
       Classification Head
               |
               ▼
         10 Class Logits
               |
               ▼
       Predicted Category
```

**Implementation Details:**

- Loads the pre-trained ViT model.
- Configures the classification layer for 10 output classes.
- Freezes the pre-trained ViT encoder parameters.
- Trains only the classification head.
- Uses AdamW optimizer with a learning rate of `0.001`.

### 2. Convolutional Neural Network (ResNet-18)

The project uses a pre-trained ResNet-18 architecture with ImageNet weights.

ResNet-18 is a CNN architecture that uses residual connections to improve information and gradient flow through deeper neural networks.

**Architecture Workflow:**

```text
          Input Image
          (224 × 224)
               |
               ▼
         Initial Convolution
               |
               ▼
          Batch Normalization
               |
               ▼
             ReLU
               |
               ▼
          Max Pooling
               |
               ▼
       Residual Block Layer 1
               |
               ▼
       Residual Block Layer 2
               |
               ▼
       Residual Block Layer 3
               |
               ▼
       Residual Block Layer 4
               |
               ▼
        Global Average Pooling
               |
               ▼
        Fully Connected Layer
               |
               ▼
          10 Class Logits
               |
               ▼
        Predicted Category
```

**Implementation Details:**

- Loads pre-trained ResNet-18 weights.
- Freezes the pre-trained convolutional parameters.
- Replaces the final fully connected layer.
- Configures the new classification layer with 10 output neurons.
- Trains only the newly initialized classification layer.
- Uses Adam optimizer with a learning rate of `0.001`.

## 🔄 Image Preprocessing

Both models use the same image preprocessing pipeline to maintain consistency during comparison.

The preprocessing operations include:

**1. Resizing**

All images are resized from 32 × 32 to 224 × 224 pixels.

**2. Tensor Conversion**

Converts images into PyTorch tensors.

**3. Normalization**

Images are normalized using ImageNet mean and standard deviation values.

Mean:

```python
[0.485, 0.456, 0.406]
```

Standard Deviation:

```python
[0.229, 0.224, 0.225]
```

Normalization helps align the input distribution with the data used to pre-train both architectures.

## 🔬 Implementation Workflow

### Step 1: Environment Setup

Install the required libraries:

```bash
pip install torch torchvision transformers scikit-learn seaborn matplotlib numpy pandas
```

### Step 2: Import Libraries

Import the required libraries for:

- Deep learning and model implementation.
- Dataset loading and preprocessing.
- Model training and evaluation.
- Performance visualization.
- Numerical operations.

The notebook automatically detects whether CUDA is available and selects GPU or CPU accordingly.

### Step 3: Dataset Loading

Load the CIFAR-10 dataset using:

```python
torchvision.datasets.CIFAR10
```

The notebook selects:

- 10,000 training images.
- 2,000 testing images.

DataLoaders are configured with:

- Training batch size: 32.
- Testing batch size: 32.
- Training shuffle: Enabled.
- Testing shuffle: Disabled.

### Step 4: Dataset Visualization

Displays a sample of 10 training images with their corresponding class labels.

This provides an initial understanding of the dataset and its image categories.

### Step 5: Vision Transformer Initialization

Load the pre-trained ViT model using:

```python
ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)
```

The original classification head is adapted to the CIFAR-10 classification task.

The Transformer encoder is frozen, and only the classification head is optimized.

### Step 6: CNN Initialization

Load the pre-trained ResNet-18 model:

```python
resnet18(weights=ResNet18_Weights.DEFAULT)
```

Freeze the existing parameters and replace the final fully connected layer:

```python
cnn_model.fc = nn.Linear(
    cnn_model.fc.in_features, 10
)
```

The new classification layer is trained using the CIFAR-10 dataset.

### Step 7: Model Training

Both models are trained using a common training function.

**Training Configuration:**

| Hyperparameter | ViT | ResNet-18 |
|---|---:|---:|
| Pre-trained Weights | Yes | Yes |
| Epochs | 3 | 3 |
| Batch Size | 32 | 32 |
| Learning Rate | 0.001 | 0.001 |
| Optimizer | AdamW | Adam |
| Loss Function | Cross Entropy | Cross Entropy |
| Feature Extractor | Frozen | Frozen |
| Classification Head | Trainable | Trainable |
| Output Classes | 10 | 10 |

The training function performs the following operations:

1. Sets the model to training mode.
2. Loads batches of images and labels.
3. Performs forward propagation.
4. Calculates Cross Entropy Loss.
5. Performs backpropagation.
6. Updates trainable parameters using the optimizer.
7. Calculates training accuracy and average loss.
8. Displays epoch-wise training performance.

### Step 8: Model Evaluation

Both models are evaluated using the same testing dataset.

The evaluation function calculates:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Average inference time per image

A classification report is also generated for all 10 categories.

### Step 9: Performance Comparison

A Pandas DataFrame is created to compare the evaluation results of ViT and ResNet-18.

A bar chart visualizes the test accuracy of both architectures.

### Step 10: Confusion Matrix

Two confusion matrices are generated:

- Vision Transformer confusion matrix.
- ResNet-18 confusion matrix.

These matrices help identify:

- Correctly classified images.
- Incorrect predictions.
- Categories frequently confused by the models.
- Class-specific classification behaviour.

### Step 11: Image Prediction

The `predict_image()` function selects an image from the testing dataset and generates predictions using both models.

The visualization displays:

- Original image.
- Actual category.
- ViT predicted category.
- ResNet-18 predicted category.

This allows direct comparison of the predictions made by both architectures on the same image.

## 📊 Evaluation Metrics

The project uses the following metrics to measure classification performance.

| Metric | Formula | Purpose |
|---|---|---|
| Accuracy | (TP + TN) / Total | Overall correct predictions |
| Precision | TP / (TP + FP) | Correctness of positive predictions |
| Recall | TP / (TP + FN) | Coverage of actual positive instances |
| F1-score | 2 × (Precision × Recall) / (Precision + Recall) | Balance between precision and recall |

The project calculates Precision, Recall, and F1-score using **Macro Averaging**, giving equal importance to each class.

### Inference Time

Inference time measures the time required by the model to generate predictions on the test dataset.

The notebook calculates average inference time per image:

\[
T_{avg}=\frac{T_{total}}{N}\times1000
\]

Where:

- \(T_{total}\) = Total evaluation inference duration in seconds.
- \(N\) = Number of test images.
- \(T_{avg}\) = Average time per image in milliseconds.

## 📁 Project Structure

```text
ViT-vs-CNN-Image-Classification/
│
├── Assignment_9.ipynb
│
├── data/
│   └── CIFAR-10 Dataset
│
├── README.md
│
└── Results/
    ├── Accuracy Comparison
    ├── Classification Reports
    └── Confusion Matrices
```

*Note: The Results directory represents the outputs generated during notebook execution. The notebook currently displays these visualizations rather than explicitly exporting them as files.*

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or above.
- Jupyter Notebook or Google Colab.
- GPU recommended for faster execution.
- Internet connection for downloading datasets and pre-trained weights.

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Navigate to the project:

```bash
cd ViT-vs-CNN-Image-Classification
```

Install dependencies:

```bash
pip install torch torchvision transformers scikit-learn seaborn matplotlib numpy pandas
```

### Run the Project

Open the notebook:

```bash
jupyter notebook Assignment_9.ipynb
```

Alternatively, upload the notebook to Google Colab and execute all cells sequentially.

## 📈 Expected Outputs

After successful execution, the notebook generates:

1. Device availability information (CPU/GPU).
2. Sample CIFAR-10 image visualization.
3. ViT model initialization.
4. ResNet-18 model initialization.
5. Epoch-wise training loss and accuracy.
6. Classification reports for both models.
7. Performance comparison table.
8. ViT vs CNN accuracy bar chart.
9. Confusion matrices for both models.
10. Side-by-side image prediction visualization.

**Note:** Actual performance scores depend on training execution and hardware configuration. No evaluation results are assumed or hardcoded in this README.

## ⚖️ ViT vs CNN: Conceptual Comparison

| Parameter | Vision Transformer | ResNet-18 CNN |
|---|---|---|
| Architecture | Transformer | Convolutional Neural Network |
| Core Mechanism | Self-Attention | Convolution |
| Feature Extraction | Image Patches | Spatial Filters |
| Context Understanding | Global attention | Hierarchical feature extraction |
| Main Advantage | Captures long-range relationships | Strong spatial feature extraction |
| Model Used | ViT-Base | ResNet-18 |
| Pre-training | ImageNet | ImageNet |
| Classification | 10 Classes | 10 Classes |

## 🔮 Future Improvements

- Train using the complete CIFAR-10 dataset.
- Experiment with more training epochs.
- Fine-tune selected Transformer encoder layers.
- Unfreeze selected CNN layers for deeper fine-tuning.
- Implement data augmentation techniques.
- Compare additional architectures such as EfficientNet and MobileNet.
- Add ROC curves and Precision-Recall curves.
- Perform hyperparameter optimization.
- Deploy the models through a Streamlit or Flask application.
- Build an interactive interface for uploading and classifying images.

## 📚 References

1. He, K., Zhang, X., Ren, S., and Sun, J. (2016). **Deep Residual Learning for Image Recognition.** CVPR.
2. Dosovitskiy, A. et al. (2021). **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.** ICLR.
3. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
4. [Hugging Face Vision Transformer Documentation](https://huggingface.co/docs/transformers/model_doc/vit)
5. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
6. [Torchvision Models Documentation](https://pytorch.org/vision/stable/models.html)
7. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<p align="center">
  <b>Comparative Study of Transformer-Based and Convolution-Based Deep Learning Architectures for Image Classification.</b>
</p>
