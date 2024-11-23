import h5py
import matplotlib.pyplot as plt
import numpy as np

def print_structure(name:str, obj:object) -> None:
    """
    Prints the name of the data as well as the objects that are associated with the data
    
    
    Args:
        name (string): Hierarchical path of the object within the HDF5 file
        obj (object): Object that is associated with the name
    """
    print(f"{name}: {obj}")


def inspect_h5_file(file_path) -> None:
    """
    Process to traverse through the object and print what's represented within the HDF5 file

    Args:
        file_path (string): The location of the ROAD_dataset.h5 file
    """
    with h5py.File(file_path, 'r') as f:
        # Walk through the file structure
        print(f)
        f.visititems(print_structure)


def inspect_train_data(file_path: str, data_group: str) -> None:
    """
    Describe what groups of data within the HPF5 file describing feature columns dimensions and data types

    Args:
        file_path (string): The location of the ROAD_dataset.h5 file
        data_group (string): The group of data interested in (test_data,train_data,anomaly_data/first_order_high_noise, anomaly_data/galactic_plane, etc..)
    """
    with h5py.File(file_path, 'r') as f:
        # Access the train_data group
        train_data_group = f[data_group]
        
        # List all datasets within train_data
        print(f"Datasets in {data_group}:")
        for dataset_name in train_data_group.keys():
            dataset = train_data_group[dataset_name]
            print(f"{dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")


def load_h5_data(file_path: str, group_name: str) -> np.array:
    """
    Loads .h5 file and separates each group into their own numpy array

    Args:
        file_path (string): file path that the .h5 data frame is stored
        group_name (string): the groups that are represented within the data (train, test, or one of the anomalies)

    Returns:
        np.array: This will output the groups of the data to an array
    """
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[group_name]['data'])
        labels = np.array(f[group_name]['labels'], dtype=str)
        ids = np.array(f[group_name]['ids'], dtype=str)
        source = np.array(f[group_name]['source'], dtype=str)
        frequency_band = np.array(f[group_name]['frequency_band'])
    return data, labels, ids, source, frequency_band


def stat_summ_print(data:np.array, labels:np.array, ids:np.array, source:np.array, frequency_band:np.array) -> None:
    """
    Outputs the prints of statistical summary of the groups of data from the different features present

    Args:
        data (np.array): Spectrograms data for a group
        labels (np.array): labels associated with the grouping
        ids (np.array): unique ids representative of each observation within a group
        source (np.array): source of where the data came from
        frequency_band (np.array): range of frequencies represented in a spectrogram for each observation
    """
    print(f"Data Shape: {data.shape}")
    print(f"Summary for Data: Min:{data.min()}, Max:{data.max()}, Mean: {data.mean()}, Std: {data.std()}")
    print(f"Summary for Frequency Band: Min:{frequency_band.min()}, Max:{frequency_band.max()}, Mean: {frequency_band.mean()}, Std: {frequency_band.std()}")
    unique_values, counts = np.unique(source, return_counts = True)
    print(f"Number of Unique Sources: {np.sum(counts)}")
    print(f"Label: {labels[0]}, ID Counts: {len(ids)}")

    
def flatten_data(data:np.array, freq_band:np.array) -> np.array:
    """
    Flattens all rows within each array to a one-dimensional array

    Args:
        data (np.array): Spectrograms data for a group
        freq_band (np.array): range of frequencies represented in a spectrogram for each observation

    Returns:
        np.array: returns array of all observations features to a flattened one-dimensional array
    """
    time_flatten = data[:, :, :, 0].flatten()        # Flatten to 1D array
    frequency_flatten = data[:, :, :, 1].flatten()
    polarization_flatten = data[:, :, :, 2].flatten()
    baseline_flatten = data[:, :, :, 3].flatten()
    freq_band = freq_band[:, :, :, 0].flatten()
    return time_flatten, frequency_flatten, polarization_flatten, baseline_flatten, freq_band


def flatten_data_index(data:np.array, freq_band:np.array, value: int) -> np.array:
    """
    Flattens a specific indexed row within each array to a one-dimensional array

    Args:
        data (np.array): Spectrograms data for a group
        freq_band (np.array): range of frequencies represented in a spectrogram for each observation
        value (int): The index of the nth observations (row) you would like to flatten

    Returns:
        np.array: returns array of nth indexed features to a flattened one-dimensional array
    """
    time_flatten = data[value, :, :, 0].flatten()        # Flatten to 1D array
    frequency_flatten = data[value, :, :, 1].flatten()
    polarization_flatten = data[value, :, :, 2].flatten()
    baseline_flatten = data[value, :, :, 3].flatten()
    freq_band = freq_band[value, :, :, 0].flatten()
    return time_flatten, frequency_flatten, polarization_flatten, baseline_flatten, freq_band


def create_pairs_plot(time_flatten: np.array, frequency_flatten:np.array, polarization_flatten:np.array, baseline_flatten:np.array, freq_band:np.array) -> None:
    """
    Creates a pair plot with each feature 

    Args:
        time_flatten (np.array): flattened time array of data
        frequency_flatten (np.array): flattened frequency of the data
        polarization_flatten (np.array): flattened polarization of data
        baseline_flatten (np.array): flattened baseline of the data
        freq_band (np.array): flattened range of frequencies represented in a spectrogram for each observation
    """
    variables = [time_flatten, frequency_flatten, polarization_flatten, baseline_flatten, freq_band]
    names = ["Time", "Frequency", "Polarization", "Station", "Frequency Band"]

    # Set up the figure grid
    fig, axes = plt.subplots(len(variables), len(variables), figsize=(12, 12))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                # Plot histogram on the diagonal
                axes[i, j].hist(var1, bins=30, color='gray', alpha=0.7)
                axes[i, j].set_title(names[i])
            else:
                # Plot scatter for pairwise combinations
                axes[i, j].scatter(var2, var1, alpha=0.5, s=1)
            
            # Set labels for rows and columns
            if j == 0:
                axes[i, j].set_ylabel(names[i])
            if i == len(variables) - 1:
                axes[i, j].set_xlabel(names[j])

    plt.tight_layout()
    plt.suptitle("Scatter Matrix of Channels", y=1.02)
    plt.show()
