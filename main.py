from data.data_preprocesing_training import DataPreprocessor
import keras
from model.model import CNNClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
import yaml

# Configurations
print('Importing Configuration Variables')
with open('./data/sampling_params.yaml', 'r') as file:
    sampling_config = yaml.safe_load(file)

h5_file_path = sampling_config['file_path']['data']
labels = sampling_config['labels']  # Replace with actual labels
percentage_contamination = sampling_config['percentage_contamination'] 
input_shape = (256, 256, 4)  # Example input shape
num_classes = len(labels)
normal_percentage = .1
total_samples = 600

print('Starting data preprocessing')

train_data, train_labels_encoded, percentages = DataPreprocessor.preprocess_lofar_data(h5_file_path, labels, percentage_contamination, normal_percentage, total_samples)
print('Completed Data Preprocessing')

print('Splitting data frame to training, validation, and test')
# Train-Test Split
# Split into train + validation (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(train_data, train_labels_encoded, test_size=0.2, random_state=42)

# Further split train + validation into train (75% of 80%) and validation (25% of 80%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Ensure labels are one-hot encoded
num_classes = len(labels)  # Number of classes

# One-hot encode labels
train_labels_onehot = to_categorical(y_train, num_classes=num_classes)
val_labels_onehot = to_categorical(y_val, num_classes=num_classes)
test_labels_onehot = to_categorical(y_test, num_classes=num_classes)

print('Training model')
# Initialize and Train CNN
#cnn = CNNClassifier(input_shape=input_shape, num_classes=num_classes)
model = CNNClassifier.build_model(input_shape, num_classes)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical crossentropy for multiclass classification
              metrics=['accuracy'])

history = CNNClassifier.train(model, X_train, train_labels_onehot, X_train_val, val_labels_onehot)
print('Model Train Complete')

print('evaluating models performance post training')
test_loss, test_accuracy = CNNClassifier.evaluate(history, X_test, test_labels_onehot)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
y_pred_probs = model.predict(X_test)  # Probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(test_labels_onehot, axis=1)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save Model
CNNClassifier.save_model(history)
print('Model weights saved')
print('Complete')
