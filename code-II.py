#C-2

import pandas as pd
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Download necessary NLTK resources (uncomment the lines below if needed)
#nltk.download('punkt')
#nltk.download('stopwords')

# Load data into separate Pandas DataFrames for train, validation, and test sets
train_data = pd.read_csv('sample_data/train.csv')
val_data = pd.read_csv('sample_data/validation.csv')
test_data = pd.read_csv('sample_data/test.csv')

# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=nltk.word_tokenize)),
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

# Print the best parameters and accuracy score
print("Best parameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)

# Use the best model to predict on the validation set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(val_data['sentence'])

# Evaluate the performance on the validation set
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(val_data['is_counterfactual'], y_pred))
print(confusion_matrix(val_data['is_counterfactual'], y_pred))

# Use the best model to predict on the test set
y_pred_test = best_model.predict(test_data['sentence'])

# Evaluate the performance on the test set
print(classification_report(test_data['is_counterfactual'], y_pred_test))
print(confusion_matrix(test_data['is_counterfactual'], y_pred_test))
