#c-3
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download necessary NLTK resources (uncomment the line below if needed)
# nltk.download('punkt')

# Load data into separate Pandas DataFrames for train, validation, and test sets
train_data = pd.read_csv('sample_data/train.csv')
val_data = pd.read_csv('sample_data/validation.csv')
test_data = pd.read_csv('sample_data/test.csv')

# Preprocessing and feature engineering
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)

# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', SVC())
])

# Define the parameter grid for optimization
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10]
}

# Perform a grid search over the parameter grid
grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(train_data['sentence'], train_data['is_counterfactual'])

# Get the best model
best_model = grid_search.best_estimator_

# Use the best model to predict on the validation set
y_pred_val = best_model.predict(val_data['sentence'])
val_accuracy = accuracy_score(val_data['is_counterfactual'], y_pred_val)

# Use the best model to predict on the test set
y_pred_test = best_model.predict(test_data['sentence'])
test_accuracy = accuracy_score(test_data['is_counterfactual'], y_pred_test)

print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)
