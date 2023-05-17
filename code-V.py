
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the training dataset
train_data = pd.read_csv('sample_data/train.csv')

# Load the validation dataset
val_data = pd.read_csv('sample_data/validation.csv')

# Load the testing dataset
test_data = pd.read_csv('sample_data/test.csv')

# Split the training data into features and labels
X_train = train_data['sentence'].values
y_train = train_data['is_counterfactual'].values

# Split the validation data into features and labels
X_val = val_data['sentence'].values
y_val = val_data['is_counterfactual'].values

# Split the testing data into features and labels
X_test = test_data['sentence'].values
y_test = test_data['is_counterfactual'].values

# Encode the labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Convert sentences to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure consistent length
max_sequence_length = max(max(map(len, X_train_seq)), max(map(len, X_val_seq)), max(map(len, X_test_seq)))
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Define the neural network model
#model = Sequential()
#model.add(Dense(64, activation='relu', input_dim=max_sequence_length))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))

# Define the neural network model
model = Sequential()
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5))  # Dropout regularization to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_data=(X_val_padded, y_val))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
