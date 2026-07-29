# Intent Classification with Transformer-based Language Models

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-green)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

This project presents an end-to-end Natural Language Processing (NLP) pipeline for **multi-class intent classification** using pre-trained Transformer models from the Hugging Face ecosystem.

The objective is to automatically classify customer support queries into one of **77 predefined intent categories**. The project follows a complete machine learning workflow, including data exploration, preprocessing, model benchmarking, hyperparameter optimization, comprehensive evaluation, and error analysis.

Three Transformer architectures were benchmarked:

* TF-IDF
* DistilBERT
* RoBERTa

After comparing their performance, the best-performing model was further optimized using **Optuna** for hyperparameter tuning. The final model achieved a **Macro F1-score of approximately 0.93**, demonstrating strong performance across all intent classes.

Beyond model training, this repository emphasizes reproducibility and project organization by separating preprocessing, training, tuning, evaluation, and visualization into independent modules, making the workflow easier to maintain and extend.

## 🔄 Workflow

```text
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Tokenization
      │
      ▼
Train / Validation Split
      │
      ▼
Model Benchmark
(DistilBERT | BERT | RoBERTa)
      │
      ▼
Best Model Selection
      │
      ▼
Hyperparameter Optimization (Optuna)
      │
      ▼
Final Training
      │
      ▼
Evaluation
      │
      ├── Metrics
      ├── Classification Report
      ├── Error Analysis
      └── Visualization
```


## 🎯 Key Takeaways

* End-to-end Transformer-based intent classification pipeline
* Benchmarking multiple pre-trained language models
* Hyperparameter optimization with Optuna
* Dynamic padding using Hugging Face DataCollator
* Comprehensive evaluation with class-wise metrics
* Error analysis through misclassified samples and confusion analysis
* Modular and production-oriented project structure

## 📂 Project Structure

```text
transformer-intent-classification/

├── data/
│   ├── raw/
│
├── models/
│
├── reports/
│   ├── metrics.json
│   ├── classification_report.csv
│   └── plots/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── metrics.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── visualize.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

##  Dataset

This project uses the **Banking77** dataset, a publicly available benchmark for intent classification in the banking domain.

The dataset consists of customer support queries collected from a mobile banking application. Each query is assigned to one of **77 predefined intent categories**, making it a challenging multi-class text classification problem.

Each sample contains:

* **text**: A customer support query written in natural language.
* **label**: A numerical class identifier.
* **label_text**: The corresponding intent name.

### Dataset Summary

| Property          | Value                             |
| ----------------- | --------------------------------- |
| Task              | Multi-class Intent Classification |
| Domain            | Mobile Banking Customer Support   |
| Total Samples     | 10,003                            |
| Number of Classes | 77                                |
| Input             | Customer support text             |
| Output            | Intent label                      |

For model development, the dataset was split into training and validation subsets using the Hugging Face `datasets` library while preserving the class distribution.

## Exploratory Data Analysis

Since this project focuses on text classification, the exploratory analysis mainly focuses on understanding the distribution of intents and the characteristics of the input text.

#### Class Distribution

The dataset contains **77 intent classes**. The class distribution was analyzed to understand potential imbalance between different intents.

Although the dataset is a multi-class classification problem, the number of samples per class is relatively limited. Therefore, class-wise evaluation metrics such as **Macro F1-score** were considered more informative than accuracy alone.

<img width="1965" height="2395" alt="class distribution" src="https://github.com/user-attachments/assets/20998453-3cc8-444e-8891-bd9ef7a13f54" />


---

#### Text Length Analysis

The length of input queries was analyzed to understand the characteristics of the text data and to determine a suitable maximum sequence length for tokenization.

The analysis showed that most customer queries are relatively short:

* Average text length: approximately **12 tokens/words**
* Median text length: approximately **10 tokens/words**
* The majority of samples are below the upper percentiles
* Only a small number of queries have significantly longer sequences

<img width="859" height="470" alt="text length" src="https://github.com/user-attachments/assets/8ec64d16-2565-4dad-819c-8c5c79a56401" />

---

#### Sequence Length Selection

Based on the text length analysis, most samples could be represented using a relatively short sequence length. Therefore, the maximum sequence length was selected to balance:

* preserving important information from longer queries
* reducing unnecessary padding
* improving computational efficiency during Transformer training

This approach helps optimize memory usage while maintaining sufficient context for intent classification.

## Data Preprocessing

Before training the Transformer models, the raw text data was transformed into a format suitable for deep learning models.

### Train/Validation Split

The dataset was divided into training and validation subsets to evaluate the generalization ability of the models during development.

The split was performed while maintaining the original class distribution to ensure that all intent categories were represented in both subsets.

---

### Tokenization

Since Transformer models cannot process raw text directly, all input queries were tokenized using the corresponding Hugging Face tokenizer for each pre-trained model.

The tokenizer converts text sequences into numerical representations:

* `input_ids`: Numerical token representations used as model input.
* `attention_mask`: Indicates which tokens are actual input tokens and which are padding tokens.

Example:

```text
Input:
"Why was my card declined?"

Output:
input_ids:
[0, 1234, 567, 89, 2]

attention_mask:
[1, 1, 1, 1, 1]
```

---

### Dynamic Padding

Instead of padding all sequences to a fixed maximum length, dynamic padding was applied using Hugging Face `DataCollatorWithPadding`.

This approach pads each batch based on the longest sequence within that batch, which:

* reduces unnecessary computation
* improves memory efficiency
* speeds up Transformer training

---

After preprocessing, the data was ready to be passed into the Transformer models for benchmarking and fine-tuning.


## 🤖 Model Benchmark

To evaluate the effectiveness of Transformer-based models, a benchmark was performed between a classical NLP baseline and pre-trained Transformer architectures.

The purpose of this comparison was to analyze the trade-off between traditional text representation methods, computational efficiency, and the ability of Transformer models to capture contextual information.

The evaluated approaches were:

* **TF-IDF Baseline**: A classical NLP approach that represents text based on word importance. This model provides a reference point to evaluate the improvement gained from Transformer-based approaches.

* **DistilBERT**: A lightweight Transformer model selected for its lower computational cost and faster training while maintaining strong language understanding capabilities.

* **Transformer-based Model (Final Candidate)**: A larger pre-trained model evaluated to determine whether additional model capacity could improve performance on the 77-class intent classification task.

All approaches were evaluated using the same validation strategy. Since the dataset contains multiple intent classes, **Macro F1-score** was selected as the primary evaluation metric to ensure that performance across all classes was considered.

### Benchmark Results

| Model                   | Accuracy | Macro F1 | Weighted F1 |
| ----------------------- | -------- | -------- | ----------- |
| TF-IDF Baseline         | 0.868    | 0.869    | 0.865       |
| DistilBERT              | 0.854    | 0.825    | 0.846       |
| Final Transformer Model | 0.907    | 0.907    | 0.906       |

The benchmark showed that Transformer-based approaches significantly improved performance compared to the classical baseline, demonstrating the advantage of contextual representations for intent classification.

## ⚙️ Hyperparameter Optimization

After selecting the best-performing Transformer architecture from the benchmark stage, hyperparameter optimization was performed to further improve the model performance.

The optimization process was conducted using **Optuna**, an efficient hyperparameter optimization framework based on Bayesian optimization strategies.

The main goal was to find a better combination of training parameters while avoiding manual trial-and-error tuning.

The following hyperparameters were optimized:

* **Learning Rate**: Controls the step size during model updates and has a significant impact on fine-tuning pre-trained Transformer models.
* **Weight Decay**: Used as a regularization technique to reduce overfitting, especially important when fine-tuning large models on relatively limited datasets.

The optimization objective was to maximize **Macro F1-score** on the validation set, as this metric provides a better evaluation of performance across all 77 intent classes.

The best configuration found by Optuna was:

| Hyperparameter | Value    |
| -------------- | -------- |
| Learning Rate  | 3.55e-05 |
| Weight Decay   | 0.098    |

Using the optimized parameters, the model achieved an improved validation Macro F1-score of:

**Best Macro F1: 0.9345**


## 📊 Model Performance

The final model was selected based on the benchmark results and further improved through hyperparameter optimization.

Since this is a **77-class intent classification** task, **Macro F1-score** was used as the primary evaluation metric to ensure that all intent categories contributed equally to the evaluation.

### Model Performance Comparison

| Model                                   | Accuracy | Macro F1   | Weighted F1 |
| --------------------------------------- | -------- | ---------- | ----------- |
| TF-IDF Baseline                         | 0.868    | 0.869      | 0.865       |
| DistilBERT                              | 0.854    | 0.825      | 0.846       |
| Transformer Model (Before Tuning)       | 0.907    | 0.907      | 0.906       |
| Transformer Model (After Optuna Tuning) | **0.936**| **0.9345** | **0.935**   |

---

### Performance Improvement

Hyperparameter optimization improved the model performance by finding a more suitable training configuration for fine-tuning the pre-trained Transformer model.

The optimized model achieved a higher Macro F1-score compared to the initial fine-tuning setup, demonstrating better performance across different intent categories.

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/a9548ea6-d674-487a-b0f5-3ff7d49d90e1" />


---

### Class-wise Evaluation

In addition to overall metrics, class-level evaluation was performed using the classification report to analyze performance across all intent categories.

This analysis helps identify:

* Strongly recognized intents
* Difficult classes with lower F1-scores
* Potentially overlapping intents causing confusion

## 🛠 Error Analysis

To better understand model behavior, a detailed error analysis was performed on the misclassified validation samples.

Instead of relying only on aggregate metrics, class-level predictions and confusion patterns were analyzed to identify challenging intent categories.

### Misclassification Analysis

The analysis showed that most model errors occurred between **semantically similar intents**, rather than completely unrelated categories.

Common confusion patterns were observed in areas such as:

* Transfer-related intents
* Top-up related intents
* Card payment issues
* Card delivery and tracking

For example, the model occasionally confused closely related categories such as:

* Pending transfer vs. transfer-related issues
* Top-up failure vs. top-up declined
* Card payment failure vs. card payment decline

These errors are expected because the corresponding intents often share similar vocabulary and require understanding subtle differences in user intent.

### Key Observations

* The model successfully learned the general domain of each query.
* Most incorrect predictions occurred between neighboring intent categories.
* There were no major patterns of unrelated class confusion.
* Limited samples per class may increase uncertainty for some low-frequency intents.

Overall, the error analysis indicates that the model learned meaningful contextual representations, while remaining challenges are mainly related to fine-grained separation between highly similar intents.

## 🚀 Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/transformer-intent-classification.git

cd transformer-intent-classification

pip install -r requirements.txt
```

The project was developed using Python and the Hugging Face ecosystem, including:

* `transformers`
* `datasets`
* `torch`
* `scikit-learn`
* `optuna`
* `evaluate`

## 🛠 Usage

The project workflow can be executed through the following steps:

### 1. Data Preparation

Prepare the dataset and configure the required paths in:

```text
src/config.py
```

### 2. Preprocessing

Run the preprocessing pipeline:

```bash
python src/preprocess.py
```

### 3. Model Training

Train the selected Transformer model:

```bash
python src/train.py
```

### 4. Hyperparameter Optimization

Run Optuna optimization:

```bash
python src/tune.py
```

### 5. Evaluation

Generate evaluation metrics and analysis reports:

```bash
python src/evaluate.py
```

### 6. Visualization

Create result plots:

```bash
python src/visualize.py
```

## 📚 Lessons Learned

Throughout this project, several practical insights were gained from building and evaluating Transformer-based NLP models:

* Evaluation metrics should match the problem characteristics. For multi-class intent classification, Macro F1 provides a better view of performance across all classes compared to accuracy alone.

* Pre-trained Transformer models require careful fine-tuning strategies. Small changes in training parameters, especially learning rate and regularization, can significantly affect performance.

* Model selection should consider both performance and computational cost. Larger models do not always provide proportional improvements.

* Error analysis is essential for understanding model behavior. Most misclassifications occurred between semantically similar intents rather than unrelated categories.

* Efficient preprocessing techniques such as dynamic padding can reduce computational overhead during Transformer training.


## 🚀 Future Work

Several improvements and extensions can be considered for future development:

* Experimenting with larger Transformer architectures and more advanced language models.
* Applying parameter-efficient fine-tuning methods such as **LoRA** and **QLoRA**.
* Performing more extensive hyperparameter optimization including training schedule parameters.
* Improving performance on low-frequency intents through data augmentation techniques.
* Deploying the model as an inference API for real-world intent classification applications.
* Extending the system toward retrieval-augmented generation (RAG) based customer support assistants.
