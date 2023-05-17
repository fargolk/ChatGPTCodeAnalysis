#Give me a python code for classification of a dataset with two columns one sentence with column name of "sentence" and the other zero or one and the column name is "is_counterfactual" with separate files for train and test and validation.
#C-1
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training dataset
train_df = pd.read_csv('sample_data/train.csv')
X_train = train_df['sentence']
y_train = train_df['is_counterfactual']

# Load the testing dataset
test_df = pd.read_csv('sample_data/test.csv')
X_test = test_df['sentence']
y_test = test_df['is_counterfactual']

# Load the validation dataset
validation_df = pd.read_csv('sample_data/validation.csv')
X_validation = validation_df['sentence']
y_validation = validation_df['is_counterfactual']

# Vectorize the sentences
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
X_validation_vectorized = vectorizer.transform(X_validation)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Predict on the validation set
y_pred_validation = model.predict(X_validation_vectorized)

# Evaluate the model on the validation set
validation_accuracy = accuracy_score(y_validation, y_pred_validation)
print("Validation Accuracy:", validation_accuracy)
