

# Fake News Prediction 

## Overview
This repository contains a Python-based system that predicts whether a news article is fake or genuine. We utilize a **Logistic Regression** model for this purpose. The system processes textual data and classifies news articles as either "fake" or "real."

## Files
- `notebook.ipynb`: A Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `train.csv`: The dataset used for training  the model.
- `test.csv`: The dataset used for testing  the model.

## Dependencies
Make sure you have the following Python libraries installed:
- pandas
- numpy
- scikit-learn


## Usage
1. **Load the Dataset**: The dataset (`train.csv`) contains labeled news articles (fake or real).
2. **Data Preprocessing**:
    - Clean the text data by removing special characters, stopwords, and irrelevant information.
    - Vectorize the text using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

3. **Train the Model**:
    - Use Logistic Regression to train the model on the preprocessed data.
4. **Evaluate the Model**:
    - Assess the model's performance using metrics such as accuracy, precision, recall, and F1-score.
5. **Predictions**:
    - Apply the trained model to new news articles to predict their authenticity.

## Results
The model achieved an accuracy of approximately 98% on the training data and 97% on the test data. You can further fine-tune hyperparameters or explore other models to improve performance.

## Future Enhancements
Consider the following improvements:
- **Feature Engineering**: Experiment with additional features (e.g., sentiment analysis, word embeddings).
- **Ensemble Methods**: Explore ensemble techniques (e.g., Random Forest, Gradient Boosting).
- **Model Explainability**: Understand which features contribute most to predictions.
- **Web Application**: Create a user-friendly web interface for users to input news articles and receive predictions.


