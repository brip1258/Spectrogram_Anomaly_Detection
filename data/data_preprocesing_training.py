import h5py
import numpy as np
class DataPreprocessor:
    def load_h5_data(h5_file_path, labels) -> np.array:
        """
        Load data from H5 file, including normal and anomaly data.
        
        Args:
            h5_file_path (str): file path location of the h5 data set
            labels (list): list of all labels associated with training
        
        Returns:
            np.array: output will be of normal data and anomaly data with associated labels from  data set
        """
        with h5py.File(h5_file_path, 'r') as hf:
            train_data = hf['train_data/data'][:]
            train_labels = hf['train_data/labels'][:].astype(str)

            # Replace '' and '1' with 'Normal'
            train_labels = np.where((train_labels == ''), 'Normal', train_labels)

            # Drop '1' labels from training data
            train_mask = train_labels != '1'
            train_data = train_data[train_mask]
            train_labels = train_labels[train_mask]

            # Load anomaly data
            anomaly_data, anomaly_labels = [], []
            for anomaly in labels:
                if anomaly != 'Normal':  # Skip 'Normal'
                    group = hf[f'anomaly_data/{anomaly}']
                    anomaly_data.append(group['data'][:])
                    anomaly_labels.extend([anomaly] * len(group['data']))

            anomaly_data = np.concatenate(anomaly_data, axis=0) if anomaly_data else np.array([])
            anomaly_labels = np.array(anomaly_labels)

        return train_data, train_labels, anomaly_data, anomaly_labels

    def subsample_normal_data(train_data: np.array, train_labels:np.array, num_normal_samples:np.array) -> np.array:
        """
        Subsample "Normal" data from the training dataset.
        Args:
            train_data (np.array): full set of training data
            train_labels (np.array): list of all labels associated with training data
            num_normal_samples: number of normal samples
        
        Returns:
            np.array: output will be of normal data with associated labels from  data set
        
        """
        normal_indices = np.where(train_labels == 'Normal')[0]
        if len(normal_indices) > num_normal_samples:
            normal_indices = np.random.choice(normal_indices, num_normal_samples, replace=False)

        other_indices = np.where(train_labels != 'Normal')[0]
        all_indices = np.concatenate([normal_indices, other_indices])
        return train_data[all_indices], train_labels[all_indices]

    def contaminate_with_anomalies(train_data: np.array, train_labels: np.array, anomaly_data: np.array, anomaly_labels: np.array, percentage_contamination: list, remaining_samples:int) -> np.array:
        """
        Add anomalies to the training dataset based on percentage contamination.
        Args:
            train_data (np.array): set of training data that will be used with normal data only
            train_labels (np.array): list of all labels associated with training data
            anomaly_data (np.array): array of data that has all occurrences of anomalous events
            anomaly_label (np.array): array of data with associated labels for anomaly data
            percentage_contamination (list): values of percentage of contaminations per category 
            remaining_samples: remainder of samples after normal calculation to append to our training data set
        
        Returns:
            np.array: output will be of normal data with associated labels from  data set
        
        """
        contaminated_data, contaminated_labels = [], []
        for anomaly, percentage in percentage_contamination.items():
            num_samples_to_add = int(remaining_samples * percentage)
            anomaly_indices = np.random.choice(
                np.where(anomaly_labels == anomaly)[0], num_samples_to_add, replace=False
            )
            contaminated_data.append(anomaly_data[anomaly_indices])
            contaminated_labels.extend([anomaly] * num_samples_to_add)

        if contaminated_data:
            contaminated_data = np.concatenate(contaminated_data, axis=0)
            contaminated_labels = np.array(contaminated_labels)

            # Append contaminated data to training data
            train_data = np.concatenate([train_data, contaminated_data], axis=0)
            train_labels = np.concatenate([train_labels, contaminated_labels], axis=0)

        return train_data, train_labels
    
    @staticmethod
    def normalize(data: np.array) -> np.array:
        """
        Normalize the data to the range [0, 1].
        
        Arg: 
            data (np.array): Sub array for training
        Returns:
            np.array: output will be a normalized sub array for training
        """
        normalized_data = np.zeros_like(data)
        for i, sample in enumerate(data):
            for channel in range(sample.shape[-1]):
                min_val, max_val = np.percentile(sample[..., channel], [1, 99])
                normalized = (sample[..., channel] - min_val) / (max_val - min_val + 1e-8)
                normalized_data[i, ..., channel] = np.clip(normalized, 0, 1)
        return normalized_data

    def shuffle_and_normalize_data(train_data: np.array, train_labels: np.array) -> np.array:
        """
        Shuffle and normalize the training data.
        
        Args:
            train_data (np.array): Sub array for training data
            train_label (np.array): Sub array for training labels
        
        Returns:
            np.array: Sub array with randomized representation of training data
        """
        shuffle_indices = np.random.permutation(len(train_data))
        train_data = train_data[shuffle_indices]
        train_labels = train_labels[shuffle_indices]

        train_data = normalize(train_data)
        return train_data, train_labels

    def calculate_class_percentages(train_labels: np.array) -> dict:
        """
        Calculate the percentages of each class in the training dataset.
        
        Args: 
            train_labels (np.array): sub array of training labels
        """
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        return {label: count / len(train_labels) for label, count in zip(unique_labels, counts)}


    def preprocess_lofar_data(
        h5_file_path: str,
        labels: list,
        percentage_contamination: list,
        seed=42,
        normal_percentage=None,
        total_samples=None
    ) -> np.array:
        """
        Preprocess LOFAR dataset for training and testing.
        
        Args:
            h5_file_path (string): file path of .h5 data
            labels (list): list of possible labels 
            percentage_contamination (list): list of percentage of anomaly contaminates
            seed (integer): value for randomization 
            normal_percentage (integer): value percentage of normal data observations
            total_samples (integer): total number of sub observations to obtain
        Return:
            np.array: returns an sub array of training data to be used for model training
        """
        np.random.seed(seed)

        # Load data
        train_data, train_labels, anomaly_data, anomaly_labels = load_h5_data(h5_file_path, labels)

        # Calculate number of "Normal" samples dynamically based on percentage
        num_normal_samples = int(total_samples * normal_percentage) if normal_percentage and total_samples else len(
            np.where(train_labels == 'Normal')[0])

        # Subsample "Normal" data
        train_data, train_labels = subsample_normal_data(train_data, train_labels, num_normal_samples)

        # Add anomalies if total_samples is specified
        if total_samples:
            remaining_samples = total_samples - num_normal_samples
            train_data, train_labels = contaminate_with_anomalies(
                train_data, train_labels, anomaly_data, anomaly_labels, percentage_contamination, remaining_samples
            )

        # Shuffle and normalize data
        train_data, train_labels = shuffle_and_normalize_data(train_data, train_labels)

        # Calculate class percentages
        percentages = calculate_class_percentages(train_labels)
        print("Class Percentages in Training Data:", percentages)

        # Encode labels
        label_mapping = {label: idx for idx, label in enumerate(labels)}
        train_labels_encoded = np.array([label_mapping[label] for label in train_labels])

        return train_data, train_labels_encoded, percentages
