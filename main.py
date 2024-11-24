from data.data_preprocesing_training import DataPreprocessor
from model.model import CNNClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import yaml

# Configurations
with open('./data/sampling_params.yaml', 'r') as file:
    sampling_config = yaml.safe_load(file)

h5_file_path = sampling_config['file_path']['data']
labels = sampling_config['labels']  # Replace with actual labels
percentage_contamination = sampling_config['percentage_contamination'] 
input_shape = (256, 256, 4)  # Example input shape
num_classes = len(labels)
normal_percentage = .1
total_samples = 600

train_data, train_labels_encoded, percentages = DataPreprocessor.preprocess_lofar_data(h5_file_path, labels, percentage_contamination, normal_percentage, total_samples)

# Train-Test Split
train_data, val_data, train_labels_encoded, val_labels_encoded = train_test_split(
    train_data, train_labels_encoded, test_size=0.2, random_state=42
)

# Initialize and Train CNN
cnn = CNNClassifier(input_shape=input_shape, num_classes=num_classes)
history = cnn.train(train_data, train_labels_encoded, val_data, val_labels_encoded)

# Save Model
cnn.save_model()
