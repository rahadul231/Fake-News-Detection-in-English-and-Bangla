# Fake News Detection in Bangla and English
### A Comparative Analysis of Classical Machine Learning and Transformer-Based Models

**Research Project | Natural Language Processing | Machine Learning | Deep Learning**

## Overview

This project presents a comparative study of automated fake news detection in **Bangla and English** using classical machine learning and transformer-based natural language processing approaches.

The research investigates how different classification models perform across two languages with different linguistic characteristics and resource availability. Separate language-specific pipelines are developed to evaluate traditional TF-IDF-based classifiers and contextual transformer models under internal validation, held-out testing, and selected external evaluation conditions.

The study uses **40,064 labeled news articles** across the two languages and additionally evaluates selected models on independent external datasets to examine cross-dataset generalization.

**Classification labels:**
- `0` — Fake News
- `1` — Real News

## Research Objectives

The principal objectives are to:

1. Develop separate fake news classification pipelines for Bangla and English.
2. Compare four classical machine learning algorithms using TF-IDF features.
3. Evaluate BanglaBERT, multilingual BERT, and XLM-RoBERTa as transformer-based classifiers.
4. Compare model performance across languages and dataset splits.
5. Investigate prediction errors, model interpretability, and statistical differences.
6. Examine cross-dataset generalization through external evaluation.

## Dataset Description

### Internal Dataset Distribution

| Language | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| English | 14,000 | 2,024 | 4,000 | 20,024 |
| Bangla | 14,000 | 2,040 | 4,000 | 20,040 |
| **Total** | **28,000** | **4,064** | **8,000** | **40,064** |

The original six internal dataset files contained 40,080 records. During quality control, 16 duplicate English validation records were removed, resulting in 40,064 records for the final internal experiment.

The training and testing datasets are class-balanced for both languages. Bangla validation is also balanced, while English validation contains 1,016 fake and 1,008 real records following duplicate removal.

### External Evaluation Datasets

| Language | External Records | Fake | Real |
|---|---:|---:|---:|
| English | 68,002 | 34,001 | 34,001 |
| Bangla | 18,150 | 9,075 | 9,075 |

These datasets provide an additional assessment of how the selected classical models perform beyond the original internal test sets.

### Dataset Preparation and Quality Control

The experiment includes:

- Missing-value, empty-text, and binary-label checks
- Within-split duplicate detection
- Exact and normalized cross-split text-overlap checks
- Whitespace normalization
- Language-specific text representations
- Consistent label mapping

No exact or normalized cross-split text overlaps were detected by the implemented checks. However, semantic duplicates, repeated news events, and source-level dependencies cannot be completely excluded.

The modeling datasets contain `text` and `label` columns. Complete row-level source provenance and annotation histories are not consistently retained in the final files.

## Methodology

### 1. Classical Machine Learning

Four classifiers are investigated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Random Forest

The models use separate **Term Frequency–Inverse Document Frequency (TF-IDF)** representations for English and Bangla.

**Main TF-IDF configuration:**

| Parameter | Value |
|---|---|
| Maximum features | 20,000 |
| N-gram range | (1, 2) |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |
| Sublinear term frequency | True |
| English lowercase | True |
| Bangla lowercase | False |

TF-IDF vectorizers are fitted exclusively on training data and then applied to the corresponding validation and testing datasets.

Linear SVM hyperparameter tuning uses three-fold cross-validation, followed by additional five-fold training-set cross-validation.

### 2. Transformer-Based Models

Three transformer architectures are evaluated:

| Transformer | English | Bangla |
|---|---|---|
| BanglaBERT | Not evaluated | Evaluated |
| Multilingual BERT (mBERT) | Evaluated | Evaluated |
| XLM-RoBERTa | Evaluated | Evaluated |

**Transformer fine-tuning configuration:**

| Parameter | Value |
|---|---|
| Training epochs | 2 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Maximum sequence length | 256 tokens |
| Random seed | 42 |

The transformer experiments use the corresponding language-specific training, validation, and testing datasets.

## Evaluation Metrics

Performance is evaluated using accuracy, precision, recall, weighted F1-score, and macro-F1. Additional analyses include ROC-AUC where applicable, confusion matrices, cross-validation, McNemar's statistical significance testing, bootstrap confidence intervals, error analysis, feature interpretation, and TF-IDF ablation experiments.

## Experimental Results

### Internal Test Performance

The following values are weighted F1-scores reported in the latest experimental notebook.

| Model | English | Bangla |
|---|---:|---:|
| Logistic Regression | 94.85% | 80.25% |
| Linear SVM | **96.45%** | 80.97% |
| Random Forest | 92.72% | 78.16% |
| BanglaBERT | — | **84.31%** |
| Multilingual BERT | 95.27% | 80.37% |
| XLM-RoBERTa | 93.60% | 80.78% |

Multinomial Naive Bayes is also included in the notebook's complete classical machine learning evaluation.

**Main observations:**

- **English:** Linear SVM achieved the highest observed internal test weighted F1-score of 96.45%.
- **Bangla:** BanglaBERT achieved the highest observed internal test weighted F1-score of 84.31%.
- Transformer models did not consistently outperform classical machine learning models across both languages.

### Validation-Based Model Selection

Logistic Regression achieved the highest classical validation weighted F1-score in both languages.

| Language | Selected Classical Model | Validation Weighted F1 |
|---|---|---:|
| English | Logistic Regression | 71.90% |
| Bangla | Logistic Regression | 82.84% |

These models were selected according to validation performance, rather than retrospectively selecting the models with the highest test scores.

### External Evaluation Results

The validation-selected Logistic Regression classifiers were evaluated on the separate external datasets.

| Language | Internal Test F1 | External F1 |
|---|---:|---:|
| English | 94.85% | 83.55% |
| Bangla | 80.25% | 61.79% |

Both models demonstrated lower weighted F1-scores on the external datasets than on the corresponding internal testing datasets.

These findings highlight the importance of cross-dataset evaluation and indicate that strong internal testing performance does not guarantee comparable performance on separately collected news.

External evaluation was performed on the selected Logistic Regression models, not on all classical and transformer architectures.

## Additional Experimental Analyses

The notebook includes several analyses beyond standard model training.

**Statistical analysis:** McNemar's tests compare paired model predictions, including comparisons between Linear SVM and Logistic Regression. Multiple-comparison adjustments are also included.

**Confidence intervals:** Bootstrap-based confidence intervals provide uncertainty estimates for selected internal testing results.

**TF-IDF ablation:** Experiments compare unigram features, unigram–bigram representations, and different feature vocabulary sizes.

**Feature interpretation:** Linear model weights are examined to identify influential words and n-grams associated with classification outcomes.

**Error analysis:** Misclassified examples are examined using descriptive, rule-based categories. These categories are exploratory rather than independently verified explanations of model errors.

## Research Findings

The experiments indicate that:

1. Linear SVM achieved the strongest observed English internal test result.
2. BanglaBERT achieved the strongest observed Bangla internal test result.
3. Validation-based model rankings differed from internal test rankings.
4. Classifier effectiveness varied across languages and datasets.
5. External evaluation revealed substantial performance reductions for the selected Logistic Regression models.

The study provides a comparative experimental framework rather than proposing a new neural network architecture.

## Limitations

The current findings should be interpreted with the following considerations:

- **English preprocessing inconsistency:** English training and testing records appear more heavily stemmed or normalized than English validation records. This may affect feature matching and model-selection reliability.
- **Incomplete provenance:** The final modeling files do not consistently preserve original article URLs, source identifiers, annotation histories, or licenses.
- **Limited external model coverage:** External evaluation was conducted for the validation-selected Logistic Regression models only.
- **Transformer verification:** mBERT checkpoint-loading warnings require further investigation.
- **Single-seed transformer experiments:** Additional runs are needed to assess training stability.
- **Input truncation:** Transformer inputs were limited to 256 tokens.
- **Remaining overlap risks:** Exact and normalized duplicate checks cannot exclude every semantically similar article or repeated news event.

The models classify news text according to dataset labels; they do not independently verify factual claims through external evidence.

## Technologies and Libraries

The project is implemented in Python using:

- **Data processing:** NumPy, Pandas
- **Classical machine learning:** Scikit-learn
- **Transformers:** Hugging Face Transformers, Datasets, PyTorch, Accelerate
- **Statistical analysis:** Statsmodels
- **Visualization:** Matplotlib, Seaborn
- **Experiment environment:** Kaggle Notebook with GPU support for transformer experiments

## Repository Contents

The main repository contains:

- `fake-news-detection.ipynb` — Complete experimental notebook
- `README.md` — Project documentation

The notebook generates evaluation tables, visualizations, and CSV outputs when executed. The source datasets and generated results are not necessarily stored in this GitHub repository.

## How to Run the Project

### Using Kaggle

1. Open `fake-news-detection.ipynb` in Kaggle.
2. Attach the required `fake-news-detection-dataset` collection containing the six internal CSV files.
3. Ensure the dataset path matches the directory configured in the notebook: `/kaggle/input/datasets/rahadul231/fake-news-detection-dataset`
4. Attach the external evaluation CSV files if you intend to reproduce the external tests.
5. Enable a GPU for transformer training.
6. Install any missing dependencies and execute the notebook cells in order.
7. Review the generated tables, evaluation metrics, and visualizations.

Transformer training may require substantial computational resources.

### Required Internal Dataset Files

The notebook expects the following CSV filenames:

- `english train.csv`
- `english validation.csv`
- `english test.csv`
- `bangla train.csv`
- `bangla validation.csv`
- `bangla test.csv`

External evaluation uses `external evaluation of English.csv` and `external evaluation of Bangla.csv`.

## Reproducibility

The experiment specifies `RANDOM_STATE = 42` and reports relevant package versions, model settings, dataset configurations, and evaluation procedures.

To reproduce the reported results, users should execute the notebook using the required dataset versions and compatible Python dependencies. The uploaded notebook does not embed all executed outputs, so the results should be regenerated and verified in the target environment.

The full workflow also depends on pretrained transformer checkpoints that must be downloaded during execution.

## Future Work

Future improvements include standardizing preprocessing across dataset splits, completing source-level provenance, checking semantic duplicates and news-event overlap, evaluating the strongest models on external datasets, verifying transformer checkpoint loading, conducting multiple-seed experiments, extending hyperparameter tuning, and testing source-aware or time-based evaluation strategies.

Further directions include model explainability, longer-context transformers, additional languages, and multimodal fake news detection.

## Academic Context

This repository supports an academic research project titled:

**Fake News Detection in Bangla and English: A Comparative Analysis of Classical Machine Learning and Transformer-Based Models**

The research focuses on language-specific classification, comparative model evaluation, and cross-dataset generalization.

The experimental results should be interpreted alongside the methodological limitations and should not be treated as proof of real-world fact-checking accuracy without additional validation.

## Author

**Md Rahadul Islam**

GitHub: [rahadul231](https://github.com/rahadul231)

## License and Data Usage

A repository license has not yet been specified.

Before redistributing datasets, pretrained weights, or derivative files, verify the licenses and usage conditions of the original data sources and model providers. Dataset source information and record-level provenance should be completed before formal research publication.
