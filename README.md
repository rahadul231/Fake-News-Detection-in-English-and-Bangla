# Fake News Detection in English and Bangla Using Classical Machine Learning and TF-IDF Features

## Project Title

**Fake News Detection in Bangla and English: A Comparative Analysis of Classical Machine Learning Models Using TF-IDF Features**

## Overview

This project presents a comparative fake news detection study for English and Bangla text using classical machine learning models and TF-IDF-based feature extraction. The main objective is to evaluate how traditional machine learning classifiers perform across a high-resource language setting, English, and a lower-resource language setting, Bangla.

The project uses cleaned, balanced, and quality-controlled binary classification datasets. The label mapping is consistent for both languages:

```text
0 = Fake News
1 = Real News
```

Separate train, validation, and test sets were used for English and Bangla to ensure fair evaluation and reduce the risk of data leakage.

## Objectives

The main objectives of this project are:

- To build a bilingual fake news detection pipeline for English and Bangla text.
- To compare classical machine learning models using TF-IDF features.
- To evaluate model performance using standard classification metrics.
- To analyze differences between English and Bangla fake news detection.
- To investigate model errors, feature importance, and language-specific challenges.
- To provide a reproducible experimental notebook for academic thesis and future research.

## Dataset Summary

The final polished datasets are binary, balanced, and cleaned.

| Language | Final Size | Fake Samples | Real Samples |
|---|---:|---:|---:|
| English | 40,022 | 20,011 | 20,011 |
| Bangla | 40,078 | 20,039 | 20,039 |

The datasets were split into:

- Training set
- Validation set
- Test set

The test set was used only for final model evaluation.

## Dataset Source and Provenance

The English and Bangla datasets were constructed from multiple publicly available fake news datasets. Before being used in the notebook, the datasets were cleaned, standardized, balanced, deduplicated, and split into train, validation, and test sets.

The final notebook uses polished dataset files containing only:

```text
text
label
```

Before final thesis or journal submission, the original dataset source names, links, licenses, collection dates, and preprocessing details should be clearly documented.

## Methodology

The overall workflow includes:

1. Dataset loading
2. Data quality checking
3. Missing value checking
4. Duplicate checking
5. Exact and normalized cross-split overlap checking
6. Exploratory data analysis
7. Text preprocessing
8. TF-IDF feature extraction
9. Model training
10. Validation-based model selection
11. Final test evaluation
12. Statistical significance testing
13. Error analysis
14. Feature interpretation
15. Ablation study
16. Comparative analysis between English and Bangla

## Feature Extraction

TF-IDF was used to convert text into numerical features.

Main TF-IDF configuration:

```text
max_features = 20000
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

Unigram and bigram features were used to capture both individual words and short word-pair patterns.

## Machine Learning Models

Four classical machine learning models were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine
4. Random Forest Classifier

Linear SVM showed strong performance across both English and Bangla datasets.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1-score
- Macro-F1
- ROC-AUC
- Confusion matrix
- Cross-validation score
- McNemar's statistical significance test

## Statistical Significance Testing

McNemar's test was used to compare whether performance differences between classifiers were statistically meaningful on the same test samples.

The notebook includes:

- Main McNemar test comparing the best model with the second-best model.
- Pairwise McNemar tests comparing Linear SVM against other models.
- Bonferroni-adjusted p-values for multiple pairwise comparisons.

## Ablation Study

A TF-IDF ablation study was included to evaluate the impact of different feature extraction settings.

The ablation experiments compared:

- Unigram only
- Unigram + bigram
- 10,000 TF-IDF features
- 20,000 TF-IDF features

The ablation was performed using Linear SVM on the validation set only. The test set was not used during ablation.

## Error Analysis

The notebook includes error analysis for both English and Bangla. Misclassified examples were reviewed to better understand model limitations.

Potential sources of error include:

- Similar writing style between fake and real news
- Overlapping political or social topics
- Emotional or sensational wording
- Bangla spelling variation
- Informal Bangla writing
- Morphological complexity
- Named-entity variation

## Bangla Language Challenges

Bangla fake news detection is more challenging than English fake news detection because Bangla text may contain:

- Rich morphology
- Spelling variation
- Informal expressions
- Mixed Bangla-English words
- Limited NLP resources
- More variation in written forms
- Topic overlap between fake and real articles

These challenges may reduce the effectiveness of TF-IDF-based models compared with more contextual models.

## Future Work

Future work may include:

- Transformer-based models such as BanglaBERT, multilingual BERT, and XLM-RoBERTa
- Larger and more diverse datasets
- External validation datasets
- Cross-lingual fake news detection
- Ensemble learning
- Explainability methods
- Multimodal fake news detection using text and images

## Limitations

This study has several limitations:

- TF-IDF features do not fully capture deep semantic meaning or sentence context.
- The results may depend on the characteristics of the source datasets.
- The notebook focuses on classical machine learning models only.
- Transformer-based models were not implemented in the main experiment.
- The study uses text-only fake news detection.
- Real-world generalization should be tested using external datasets.

## Files in This Repository

Recommended repository structure:

```text
.
├── fake-news-detection-in-english-and-bangla.ipynb
├── README.md
├── data/
│   ├── english_train.csv
│   ├── english_validation.csv
│   ├── english_test.csv
│   ├── bangla_train.csv
│   ├── bangla_validation.csv
│   └── bangla_test.csv
├── results/
│   ├── validation_results.csv
│   ├── final_test_results.csv
│   ├── final_summary_table.csv
│   ├── best_hyperparameters.csv
│   ├── cross_validation_summary.csv
│   ├── confusion_matrix_summary.csv
│   ├── pairwise_mcnemar_results.csv
│   ├── english_top_tfidf_features.csv
│   ├── bangla_top_tfidf_features.csv
│   ├── english_error_analysis.csv
│   ├── bangla_error_analysis.csv
│   └── tfidf_ablation_results.csv
└── requirements.txt
```

## Requirements

Main Python libraries used:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
```

You can install the required packages using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/rahadul231/Fake-News-Detection-in-English-and-Bangla.git
```

2. Open the notebook:

```text
fake-news-detection-in-english-and-bangla.ipynb
```

3. Upload or place the dataset files in the correct directory.

4. Run all cells from top to bottom.

5. The notebook will generate evaluation tables, plots, statistical tests, error analysis, and result CSV files.

## Reproducibility

A fixed random seed was used:

```python
RANDOM_STATE = 42
```

The notebook also reports Python and package versions to support reproducibility.

## Academic Use

This notebook was developed as part of an academic thesis project on bilingual fake news detection. It can be used as a baseline experimental pipeline for future research on English and Bangla fake news classification.

For thesis or journal submission, users should include:

- Full dataset source citations
- Dataset license information
- Ethical/data-use statement
- Literature review
- Comparison with recent published studies
- External validation or transformer-based extension if required

## Citation

If you use this project, please cite the repository or related thesis/manuscript.

```text
Md Rahadul Islam. Fake News Detection in Bangla and English: A Comparative Analysis of Classical Machine Learning Models Using TF-IDF Features.
```

## Author

**Md Rahadul Islam**

## License

Add your selected license here.

Recommended options:

- MIT License, if you want others to freely use and modify the code.
- CC BY 4.0, if you mainly want to share academic content with attribution.
- Custom license, if dataset redistribution is restricted.

Important: Make sure your dataset licenses allow redistribution before uploading the dataset files publicly.
