import os
import datetime
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


class CNNClassifier:
    def build_model(input_shape: tuple, num_classes: int) -> keras.models.Sequential:
        """
        Builds the CNN model with the specified activation function.
        Args:
            input_shape (tuple): represents the height, width, and channels
            num_classes (int): number of classes total
        Returns:
            keras.models.Sequential: compiled CNN model 
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')  # Softmax for multiclass classification
        ])
        return model

    def train(model: keras.models.Sequential, train_data: np.array, train_labels: np.array, val_data: np.array, val_labels: np.array) -> keras.models.Sequential:
        """
        Train the CNN model with customizable batch size and early stopping.
        Arg:
            model (keras.models.Sequential): model to train
            train_data (np.array): Training data
            train_labels (np.array): Training labels (one-hot encoded)
            val_data (np.array): Validation data
            val_labels (np.array): Validation labels (one-hot encoded)
        Return: 
            keras.models.Sequential model
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',       # Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            patience=3,               # Number of epochs with no improvement after which training will be stopped
            verbose=1,                # Verbosity level
            restore_best_weights=True # Restore the model weights from the epoch with the best monitored value
        )
        history = model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=10,
            batch_size=32,
            callbacks=[early_stopping]
        )

        return history

    def evaluate(model: keras.models.Sequential, test_data: np.array, test_labels: np.array, batch_size=32) -> int:
        """
        Evaluate the model on test data.
        Args:
            model (keras.models.Sequential): model to train
            test_labels: Test labels (one-hot encoded)
            batch_size: Batch size for evaluation
        Return: 
            Test loss and accuracy as integers
        """
        test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
        return test_loss, test_accuracy

    def predict(model: keras.models.Sequential, data: np.array):
        """
        Make predictions using the model.
        Args:
            model (keras.models.Sequential): CNN model
            data (np.array): Data for prediction
        Return:
            Predicted class probabilities
        """
        return model.predict(data)

    def save_model(model: keras.models.Sequential, directory="./model_weights") -> None:
        """
        Save the trained model weights.
        Args:
            model (keras.models.Sequential): trained model
            directory (string): Directory to save the model weights
        """
        os.makedirs(directory, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(directory, f"model_weights_{current_time}.h5")
        model.save_weights(file_name)
        print(f"Model weights saved to {file_name}")

    def save_history(history: keras.models.Sequential, directory="./model_weights")-> None:
        """
        Save the training history as a pickle file.
        Args:
            model (keras.models.Sequential): trained model
            directory (string): Directory to save the model weights
        """
        os.makedirs(directory, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(directory, f"training_history_{current_time}.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(history.history, file)
        print(f"Training history saved to {file_name}")

    def calculate_metrics(y_true: np.array, y_pred_probs: np.array) -> int:
        """
        Calculate precision, recall, and F1-score.
        Arg:
            y_true (np.array): Ground truth labels (encoded as integers)
            y_pred_probs (np.array): Predicted probabilities
        Return: 
        integer values of Precision, Recall, and F1-score
        """
        y_pred = np.argmax(y_pred_probs, axis=1)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return precision, recall, f1
