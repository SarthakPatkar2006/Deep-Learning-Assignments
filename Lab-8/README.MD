# 🤖 BERT-Based Sentiment Analysis

<p align="center">
  <b>Deep Learning | Natural Language Processing | Transformer Models</b>
</p>

<p align="center">
  A Deep Learning project that fine-tunes a pre-trained BERT model to classify movie reviews into Positive and Negative sentiments.
</p>

---

## 📌 Project Overview

This project implements a **BERT-based Sentiment Analysis System** using Transfer Learning and Natural Language Processing (NLP).

The system leverages the pre-trained `bert-base-uncased` Transformer model and fine-tunes it on a sample of the **IMDB Movie Reviews Dataset** to perform binary sentiment classification.

Given a movie review as input, the model predicts whether the expressed sentiment is Positive or Negative and provides a confidence score for its prediction.

The implementation uses the Hugging Face Transformers library, PyTorch, and Scikit-learn for model training, inference, and evaluation.

## 🎯 Objectives

- Understand the architecture and working of BERT.
- Implement transfer learning using a pre-trained Transformer model.
- Perform text preprocessing and tokenization.
- Fine-tune BERT for binary sentiment classification.
- Evaluate model performance using standard classification metrics.
- Generate sentiment predictions with confidence scores.
- Save the fine-tuned model for future inference.

## ✨ Key Features

- **Pre-trained BERT Model:** Uses `bert-base-uncased` for contextual text representation.
- **Transfer Learning:** Fine-tunes an existing language model instead of training from scratch.
- **Text Tokenization:** Converts raw movie reviews into BERT-compatible token representations.
- **Binary Classification:** Classifies reviews as Positive or Negative.
- **Model Evaluation:** Calculates Accuracy, Precision, Recall, and F1-score.
- **Confidence Estimation:** Uses Softmax probabilities to determine prediction confidence.
- **Batch Prediction:** Supports sentiment analysis for multiple text inputs.
- **Model Saving:** Saves the fine-tuned model and tokenizer for future use.

## 🧠 Model Architecture

The project uses the BERT (Bidirectional Encoder Representations from Transformers) architecture with a sequence classification head.

### Architecture Workflow

```text
             Input Movie Review
                     |
                     ▼
              Text Preprocessing
                     |
                     ▼
              BERT Tokenizer
                     |
                     ▼
       Input IDs + Attention Mask
                     |
                     ▼
          Pre-trained BERT Model
             (bert-base-uncased)
                     |
                     ▼
           Contextual Embeddings
                     |
                     ▼
          Classification Head
                     |
                     ▼
               Output Logits
                     |
                     ▼
              Softmax Function
                     |
                     ▼
            Sentiment Prediction
                     |
              ┌──────┴──────┐
              ▼             ▼
           Negative       Positive
              |             |
              └──────┬──────┘
                     ▼
              Confidence Score
```

### How BERT Works

BERT is a Transformer-based language model that understands the context of words by processing text bidirectionally.

Unlike traditional approaches that process words independently or in a fixed direction, BERT considers both left and right context when generating contextual representations.

The model consists of:

1. **Embedding Layer:** Converts input tokens into numerical representations.
2. **Transformer Encoder:** Captures contextual relationships using self-attention mechanisms.
3. **Contextual Representation:** Generates meaningful representations of the input sequence.
4. **Classification Head:** Produces logits for the two sentiment classes.
5. **Softmax Layer:** Converts logits into class probabilities.

## 📂 Dataset

The project uses the **IMDB Movie Reviews Dataset** available through Hugging Face Datasets.

**Dataset:** `stanfordnlp/imdb`

The dataset contains movie reviews with corresponding sentiment labels.

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

### Dataset Configuration

For faster execution during experimentation, a subset of the original dataset is selected.

| Parameter | Value |
|---|---:|
| Training Samples | 2,000 |
| Testing Samples | 500 |
| Classification Type | Binary |
| Number of Classes | 2 |
| Random Seed | 42 |

The training and testing samples are shuffled using a fixed random seed to maintain reproducibility.

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | BERT implementation and fine-tuning |
| Hugging Face Datasets | Dataset loading and processing |
| Scikit-learn | Evaluation metrics |
| NumPy | Numerical computations |
| Google Colab | Notebook execution environment |

## 🔄 Implementation Workflow

### 1. Environment Setup

Install the required dependencies:

```bash
pip install -U transformers datasets accelerate scikit-learn
```

### 2. Dataset Loading

Load the IMDB dataset using Hugging Face Datasets.

- Load training and testing datasets.
- Shuffle the datasets using a fixed seed.
- Select 2,000 training samples and 500 testing samples.

### 3. Text Tokenization

The pre-trained BERT tokenizer converts raw text into numerical representations.

The implementation uses:

- `AutoTokenizer`
- Truncation enabled
- Maximum sequence length of 256 tokens
- Dynamic padding through `DataCollatorWithPadding`

Tokenization generates input IDs and attention masks required by BERT.

### 4. Model Initialization

Load the pre-trained model:

`bert-base-uncased`

The model is initialized using:

`AutoModelForSequenceClassification`

The classification layer is configured with two output classes:

- Class 0: NEGATIVE
- Class 1: POSITIVE

### 5. Model Fine-Tuning

The pre-trained BERT model is fine-tuned using labelled IMDB reviews.

**Training Configuration:**

| Hyperparameter | Value |
|---|---:|
| Pre-trained Model | bert-base-uncased |
| Epochs | 1 |
| Training Batch Size | 8 |
| Evaluation Batch Size | 16 |
| Learning Rate | 0.00002 |
| Weight Decay | 0.01 |
| Maximum Sequence Length | 256 |
| Random Seed | 42 |
| Evaluation Strategy | Epoch |
| Optimizer | Trainer default |

The Hugging Face `Trainer` API manages the training and evaluation process.

### 6. Model Evaluation

The trained model is evaluated on the testing dataset.

The following performance metrics are calculated:

- **Accuracy:** Measures the proportion of correctly classified reviews.
- **Precision:** Measures how many predicted positive instances are actually positive.
- **Recall:** Measures how many actual positive instances are correctly identified.
- **F1-Score:** Harmonic mean of Precision and Recall.

A detailed classification report is also generated using Scikit-learn.

### 7. Sentiment Prediction

The trained model is tested on new movie-review sentences.

The prediction pipeline:

1. Accepts one or multiple text inputs.
2. Tokenizes the input sentences.
3. Generates model logits.
4. Applies the Softmax function.
5. Selects the class with the highest probability.
6. Returns the predicted sentiment and confidence score.

### 8. Model Saving

The fine-tuned model and tokenizer are saved locally using:

```python
trainer.save_model("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
```

The saved model can be loaded later without repeating the training process.

## 📊 Evaluation Metrics

The project evaluates model performance using four standard classification metrics.

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) |

Where:

- TP = True Positive
- TN = True Negative
- FP = False Positive
- FN = False Negative

**Note:** Actual evaluation scores are generated when the notebook is executed. They are not hardcoded in this documentation.

## 📁 Project Structure

```text
BERT-Sentiment-Analysis/
│
├── Assignemt_8.ipynb
│
├── bert_sentiment_results/
│   └── Training checkpoints and evaluation outputs
│
├── bert_sentiment_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── Other saved model files
│
└── README.md
```

*Note: The model directories are generated after executing the training and saving cells.*

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or above
- Google Colab or Jupyter Notebook
- GPU recommended for faster model fine-tuning
- Internet connection for downloading the dataset and pre-trained model

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd BERT-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -U transformers datasets accelerate scikit-learn
```

### Run the Project

Open the notebook:

```bash
jupyter notebook Assignemt_8.ipynb
```

Alternatively, upload the notebook to Google Colab and execute the cells sequentially.

## 💡 Sample Predictions

The notebook tests the model using the following sample sentences:

| Input Review | Expected Sentiment |
|---|---|
| This movie was absolutely fantastic. I loved every minute of it! | Positive |
| The story was boring and the acting was terrible. | Negative |
| It was an enjoyable movie with excellent performances. | Positive |

The model generates its own predicted sentiment and confidence score during execution.

## 🔬 Important Deep Learning Concepts

This project demonstrates several important concepts:

**1. Transfer Learning**

Reusing knowledge from a pre-trained model and adapting it to a specific downstream task.

**2. Fine-Tuning**

Updating the pre-trained BERT parameters using a labelled sentiment classification dataset.

**3. Attention Mechanism**

Allows the Transformer model to focus on relevant tokens while understanding contextual relationships.

**4. Tokenization**

Converts textual input into token IDs that can be processed by the neural network.

**5. Softmax Function**

Converts classification logits into probability distributions.

\[
P(y_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

**6. Classification Head**

Maps the contextual representation produced by BERT into two sentiment classes.

## ⚠️ Limitations

- The model is trained on a subset of the original IMDB dataset.
- Training is performed for only one epoch to reduce computational requirements.
- Maximum input sequence length is limited to 256 tokens.
- Performance may vary depending on the training sample and execution environment.
- The model is primarily designed for English movie reviews and may not generalize equally well to other domains or languages.

## 🔮 Future Improvements

- Increase the training dataset size.
- Experiment with multiple training epochs.
- Compare BERT with DistilBERT, RoBERTa, and other Transformer architectures.
- Implement a web-based sentiment analysis interface.
- Add sentiment visualization and prediction history.
- Evaluate model performance using a confusion matrix and ROC-AUC.
- Experiment with hyperparameter optimization.

## 📚 References

1. Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** NAACL-HLT.
2. [BERT – Hugging Face Model](https://huggingface.co/google-bert/bert-base-uncased)
3. [IMDB Dataset – Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)
4. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
5. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
6. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<p align="center">
  <b>Built as a Deep Learning and Natural Language Processing implementation project.</b>
</p>