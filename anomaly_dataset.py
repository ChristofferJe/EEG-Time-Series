import numpy as np
import torch
import torch.utils.data as tud
import bidirectional_dataset
from scipy import signal


class AnomalyDataset(torch.utils.data.Dataset):
    """Dataset based on the CutoutDataset. Simply takes the sections from
    the CutoutDataset and inserts some toy anomalies 
    (e.g. square waves or sine waves)
    """

    def __init__(self, cutout_dataset : bidirectional_dataset.CutoutDataset, i,
                 anomaly_duration = 15, anomaly_magnitude = 1):
        """
        Args:
            cutout_dataset: A CutOut dataset to insert anomalies into
            i: Index which lets the dataset know the position in the list
                of all subdatasets. Used for setting the random number generator
                seed (ensures deterministic results)
            anomaly_duration: The duration in time steps of the anomaly (default 15)
            anomaly_magnitude: Magnitude of the anomaly (default 1)
        """
        self.cutout_dataset = cutout_dataset
        self.index = i 
        self.anomaly_duration = anomaly_duration
        self.anomaly_magnitude = anomaly_magnitude


    def __len__(self):
        return len(self.cutout_dataset)


    def get_n_channels(self):
        '''Returns the number of channels '''
        return self.cutout_dataset.channel_names.get_n_channels()


    def section_get_points(self, idx):
        '''For section with index idx, get time step for:
        - Start of section
        - End of section
        '''
        return self.cutout_dataset.section_get_points(idx)


    def get_channel_names(self):
        '''get the channel names'''
        return self.cutout_dataset.get_channel_names()

    
    def __getitem__(self, idx):
        '''Returns before and after a target and the target
        '''
        # retrieve the data from the cutout dataset
        (before, after_flipped) , (target, target_flipped) = self.cutout_dataset[idx]

        # decide if sections should have anomaly
        seed = self.index * 1000000 + idx
        rng = np.random.default_rng(seed)
        has_anomaly = (rng.random() < 0.25) ## 25% chance of anomaly in section
        # join the series up to anomalize it (so that anomalies may appear 
        # across cutout border)
        
        if has_anomaly:
            # stitch back together the sequence
            joined = np.concatenate((before, target, np.flip(after_flipped, axis=1)), axis=1)
            # loop over channels and anomalize
            for channel in range(joined.shape[0]):
                joined[channel, :] = anomalize(joined[channel, :],
                    seed, self.anomaly_duration, self.anomaly_magnitude)             
            # split up the anomalized data once again
            before = joined[:, :before.shape[1]]
            target = joined[:, before.shape[1]:before.shape[1] + target.shape[1]]
            target_flipped = np.copy(target[:, ::-1])
            after_flipped = np.copy(np.flip(joined[:, before.shape[1] + target.shape[1]:], axis=1))
        return (before, after_flipped) , (target, target_flipped), has_anomaly
    

def create_combined_dataset(proportion_to_use = 1.0, time_before_cutout = 1,
                            time_after_cutout = 1, cutout_duration = 1,
                            subjects = ['007'] , channel_names = ['ELA', 'ELE', 'ELI', 'ERA', 'ERG', 'ERE', 'ERI'],
                            resample_freq = 50, standardization_info = None):
    '''Create a combined anomaly dataset for multiple subjects
    Args:
    proportion_to_use: Proportion of the total sleep sequence to include (if 
        below 1, will cut off the end)
    time_before_cutout: Time in seconds to include before the cutout in each example
    time_after_cutout: Time in seconds to include after the cutout
    cutout_duration: Time in seconds of the cutout duration
    subjects: List of string identifiers of subjects to include in the dataset
    channel_names: List of the channel names to include
    resample_freq: The desired frequence to resample the data to
    standardization_info: If unspecified, the data is standardized globally across
        all subjects so that each channel has mean 0 and std 1.
        If a dictionary {"mean": M, "std": S} is specified, the data will instead
        be subtracted by M and divided by S (this is used to normalize the
        validation/test set using the data from the training set)
    Returns:
        tuple (combined_dataset, individual_datasets, standardization_info)
        where
        combined_dataset is a torch.utils.data.Dataset containg all
            the subjects (can be fed to a dataload)
        individual_datasets is a list of the individual AnomalyDatasets for each
            subject
        standardization_info is a dictionary {"mean": M, "std": S} including
            the information that was used to standardize this dataset
    '''
    
    # get some CutoutDatasets that we can anomalize
    _, cutout_datasets, new_standardization_info = bidirectional_dataset.create_combined_dataset(
        proportion_to_use, time_before_cutout, time_after_cutout, cutout_duration,
        subjects, channel_names, resample_freq, standardization_info
    )

    # now turn into AnomalyDatasets
    anomaly_datasets = [AnomalyDataset(dataset, i) for i, dataset in enumerate(cutout_datasets)]
    
    # Concatenate all the datasets
    # also return individuals datasets along with standardization info
    return tud.ConcatDataset(anomaly_datasets), anomaly_datasets, new_standardization_info


def anomalize(timeseries, seed, anomaly_duration=100000, anomaly_magnitude=10,\
              min_dist=0, given_function=None, a_types=5, spread=0.25):
    """
    Inserts anomalies into a time series.

    Parameters:
    - time_series: The input time series. Expects an array: array([0.476,  -6.229, ...,  -3.603, -12.586]) 
    - seed: The seed to set (ensures deterministic results across an entire dataset)
    - anomaly_duration: The duration of each anomaly.
    - anomaly_magnitude: The magnitude of each anomaly.
    - min_dist: The minimum distance between anomalies as well as to the start and end of the series.
    - given_function: Choose a function to be the anomaly.
    - a_types: number of types of diferrent anomalies that can be aplied see ##BELOW## 
    - spread: Changes the spread of the x-axis such that the applied functions become more spread out.

    Returns:
    - A tuple (T, L), where T is the time series with anomalies and L is a binary vector indicating anomaly locations.
    """
    # Assertions: 
    assert anomaly_duration > 0, "anomaly_duration should be greater than 0"
    assert anomaly_magnitude >= 0, "anomaly_magnitude should be greater than or equal to 0"
    assert min_dist >= 0, "min_dist should be greater than 0 or equal to 0"

    length = len(timeseries)
    anomaly_locations = [] 

    rng = np.random.default_rng(seed)
    
    # Finding locations for the anomalies:
    anomaly_locations = [rng.integers(0, length-anomaly_duration)]


    # 0-1 anomaly vector:
    time_series_with_anomalies = np.copy(timeseries)
    x = np.linspace(0.00001, int(np.sqrt(anomaly_duration)/spread), anomaly_duration) # for the functions (might need to be adjusted)
    K = max(abs(timeseries))
    
    for loc in anomaly_locations:
        start = max(0, loc) # max og min not neccesery but I put them in none the less
        end = min(length, loc + anomaly_duration)
        C = anomaly_magnitude*K


        if given_function is None:
            option = rng.integers(0, min(a_types,5))
            if option == 0:
                time_series_with_anomalies[start:end] += signal.square(x) * C
            if option == 1:
                time_series_with_anomalies[start:end] += np.sin(x) * C
            if option == 2:
                time_series_with_anomalies[start:end] += np.sin(x)/x * C
            if option == 3:
                time_series_with_anomalies[start:end] += np.sqrt(x)/3 * C
            if option == 4:
                time_series_with_anomalies[start:end] += np.repeat(np.mean(timeseries[start:end]), anomaly_duration)
        else:
            time_series_with_anomalies[start:end] = given_function(x) * C
    return time_series_with_anomalies