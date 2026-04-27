# Fake Job Posting Detection

A machine learning project under supervision which takes in structured attributes, text attributes and classification algorithms to classify job advertisements as either **real** or as a fraud.

## Project Overview

Through online job sites, employers and job seekers are easily linked to each other but this also brings about the chances of fraudulent job advertisements. Such bogus listings can be employed in scams, phishing, or personal information gathering of applicants.

The objective of the given project is to construct and test machine learning models which will determine automatically whether a job posting is legitimate or not. It is a binary problem of a supervised learning.

## Dataset

The datasource of this project is the Kaggle dataset of **Real or Fake Job Posting Prediction**:

[Dataset Link](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

### Dataset Summary
- Job posting records are contained in it.
- It has both text features and categorical/structured features.
- Target variable reports on the posting as being either a **fraudulent or real posting.

## Objectives

- Process and clean the data.
- Conduct the exploratory data analysis (EDA).
- Deal with missing data and disproportion.
- Seek information that is useful in text fields.
- Train various monitored machine learning models.
- Evaluate model performance through evaluation measures.
- Find the best method of fake job detection.

## Machine Learning Task

- **Learning Type Supervised Learning.
- **Type of problem: Binary Classification.
- **The Target Variable: Fraudulent vs Real job posting.

## Methods

The following steps and methods might be present in the project:

### Data Preprocessing
- Missing value handling
- Coding nominal or ordinal variables.
- Text preprocessing
- Selection of features / feature engineering.
- Train-validation-test split

### Exploratory Data Analysis
- Distribution analysis of class distribution.
- Missing data inspection
- Relationship feature analysis.
- Important variables visualization.
- Pattern exploration by use of text.

### Models
Potential models to be tested are:
- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine
- Gradient Boosting / XGBoost
- TF-IDF-based vectorization of NLP.

## Evaluation Metrics

The accuracy is not sufficient because fraudulent postings are not normally as numerous as the real ones. The metrics determined in this project are:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Repository Structure

```text
fake-job-posting-detection/
│
\{\|human|>data/         # Files or notes on data access.
├── notebooks/             # EDA and model notebooks.
├── src/                   # Python code of preprocessing and training.
├── reports/               # Final report and documents.
├── presentation/    # Presentation materials and Slides.
├── README.md              # Project overview
└── requirements.txt       # Project requirements.
