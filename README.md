#  Mental Health Classification

Multi-class NLP system that detects mental health conditions from social media text using classical ML and deep learning.

## Overview

This project classifies short social media statements into three mental health categories: **Normal**, **Depression** and **Suicidal**  using Natural Language Processing (NLP). It explores and compares multiple machine learning models, from baseline Logistic Regression to a fine-tuned BERT transformer to identify the most effective classifier.


## Dataset

**Source:** [Sentiment Analysis for Mental Health – Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

- Social media posts labelled across multiple mental health categories
- Filtered to the **top 3 classes**: Normal, Depression, Suicidal
- Balanced to ~2,500 samples per class using upsampling
- Preprocessing: lowercasing, punctuation removal, stopword removal, lemmatisation (NLTK)


## Project Structure

```
Mental-Health-Classification/
├── Mental_Health_Classification.ipynb      # Full pipeline: EDA → preprocessing → modelling → evaluation
├── 3_Mental_Health_Classification.ipynb    # 3-class focused version with ensemble & BERT
└── README.md
```

---

## Methodology

### Text Preprocessing
- Contraction expansion
- Lowercasing & punctuation removal
- Stopword removal (NLTK)
- Lemmatisation (WordNetLemmatizer)

### Feature Extraction
- **TF-IDF Vectorisation** (5,000 and 10,000 features)

### Models Trained & Compared

| Model | Accuracy |
|---|---|
| Logistic Regression (Baseline) | ~79% |
| Logistic Regression (Enhanced, C=10) | 80.22% |
| LinearSVC / SVM (C=0.5) | 80.22% |
| Voting Ensemble (LR + SVM) | 80.25% |
| **Stacking Ensemble** | **80.68%  Best** |
| BERT (fine-tuned) | - |

### Best Model: Stacking Ensemble
A meta-learner stacking Logistic Regression and SVM achieved the highest accuracy of **80.68%**, with strong ROC AUC scores (>0.90) across all three classes.

---

## Results

- **Stacking Ensemble** selected as best model based on accuracy and AUC
- ROC curves generated for Enhanced LR, Smart SVM, and Stacking
- Precision-Recall curves confirm strong performance especially for Suicidal class detection
- Top TF-IDF features visualised via coefficient analysis

---

##  How to Run

1. Open the notebook in **Google Colab**
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update the dataset path in the notebook to your Drive location
4. Install any missing dependencies:
   ```bash
   pip install contractions wordcloud transformers datasets
   ```
5. Run all cells in order — preprocessing → training → evaluation → interactive prediction

---

##  Interactive Prediction

The notebook includes an **interactive prediction mode** where you can type any statement and get real-time classification from three models (LR, SVM, Ensemble), along with model agreement scores and an interpretation.

```
Input: "I feel hopeless and don't want to go on"
→ LR:       Depression
→ SVM:      Suicidal
→ Ensemble: Suicidal
Model Agreement: 66% | Majority: Suicidal
 **Urgent: Shows suicidal thoughts. Seek immediate help.**-
```

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas, numpy | Data handling |
| NLTK | Text preprocessing |
| scikit-learn | ML models & evaluation |
| HuggingFace Transformers | BERT fine-tuning |
| matplotlib, seaborn | Visualisation |
| WordCloud | Feature visualisation |
| Google Colab | Runtime environment |

---
## Author

**Jeshmin Shrestha**
