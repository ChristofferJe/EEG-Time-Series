import os
import os.path as op
import numpy as np
import torch
import torch.utils.data as tud
from mne_bids import (
    BIDSPath,
    read_raw_bids,
)


# ---------------- ORIGINAL DATASET ---------------- #

class CutoutDataset(torch.utils.data.Dataset):
    """Dataset which returns time series in multipe channels before and after some target

    """

    def __init__(self, series, step_start, step_stop, time_before_cutout = 1,
                 time_after_cutout = 1, cutout_duration = 1,
                 channel_names = ['ELA', 'ELE', 'ELI', 'ERA', 'ERG', 'ERE', 'ERI'],
                sfreq = 20):
        """
        Args:
            step_start: The first step to be included in this dataset
            step_stop: The last step to be included in this dataset
            time_before_cutout: Time (in seconds) to include before the cutout
            time_after_cutout: Time (in seconds) to include after the cutout
            cutout_duration: Duration (in seconds) of the cutout
            channel_names: The names of the channels to load
            sfreq: The desired sampling frequency
        """

        self.series = series
        self.sfreq = sfreq
        
        # setup cutout stuff
        self.step_start = step_start
        self.step_stop = step_stop
        self.steps_before_cutout = int(time_before_cutout * self.sfreq)
        self.steps_after_cutout = int(time_after_cutout * self.sfreq)
        self.cutout_duration = int(cutout_duration * self.sfreq)
        self.channel_names = channel_names

        self.total_steps = step_stop - step_start
        
        self.section_duration = self.steps_before_cutout + self.steps_after_cutout + self.cutout_duration
        self.no_sections = int((
            self.total_steps - (self.steps_before_cutout + self.steps_after_cutout)
        ) // self.cutout_duration)

    def __len__(self):
        return self.no_sections


    def get_n_channels(self):
        '''Returns the number of channels '''
        return len(self.channel_names)


    def section_get_points(self, idx):
        '''For section with index idx, get time step for:
        - Start of section
        - End of section
        '''
        section_start = int(self.cutout_duration * idx)
        section_end = int(section_start + self.section_duration)

        return section_start, section_end


    def get_channel_names(self):
        '''get the channel names'''
        return self.channel_names

    
    def normalize(self, mean, std):
        '''Normalize the data given a mean and std
        We pass them this way in order to normalize globally (across multiple subjects'''
        self.series -= (self.series - mean[:, np.newaxis]) / std[:, np.newaxis]
                  
    
    def __getitem__(self, idx):
        '''Returns tuple
        (before, after_flipped) , (target, target_flipped)
        where before is the stuff before the cutout
            after_flipped is the stuff after the cutout (flipped)
            and target and target_flipped are unflipped and flipped versions
            of the target (what really is in the cutout)
        '''
        section_start, section_end = self.section_get_points(idx)
        
        # note when indexing: section_start is the new 0
        series = self.series[:, section_start:section_end] #shape: (n_channels, sequence_len)
        cutout_start = self.steps_before_cutout
        cutout_end = cutout_start + self.cutout_duration

        # Get the stuff before the hole 
        before = series[:,:cutout_start] # shape: (n_channels, sequence_len)
        # Get the stuff after the hole and flip
        after_flipped = np.copy(np.flip(series[:,cutout_end:], axis = 1)) #shape: (n_channels, sequence_len)
        target = series[:,cutout_start:cutout_end] #shape: (n_channels, sequence_len)
        target_flipped = np.copy(np.flip(target, axis = 1)) #shape: (n_channels, sequence_len)

        return (before, after_flipped) , (target, target_flipped)


def create_combined_dataset(proportion_to_use = 1.0, time_before_cutout = 1,
                            time_after_cutout = 1, cutout_duration = 1,
                            subjects = ['007'],
                            channel_names = ['ELA', 'ELE', 'ELI', 'ERA', 'ERG', 'ERE', 'ERI'],
                            resample_freq = 20,
                           standardization_info=None):
    '''Create a combined dataset for multiple subjects
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
        individual_datasets is a list of the individual CutoutDatasets for each
            subject
        standardization_info is a dictionary {"mean": M, "std": S} including
            the information that was used to standardize this dataset
    '''

    # define some generic stuff for loading the proper files
    dataset = "ds004348"
    session = "001"
    datatype = "eeg"
    extension = ".set"
    task = "sleep"
    path = "data/eesm17" #assumes data has been downloaded from https://openneuro.org/datasets/ds004348
    
    # LOAD DATA
    # Define the folder where the data is stored
    bids_root = path
    if not op.isdir(bids_root):
        os.makedirs(bids_root)

    # list to hold the timeseries data for each subject
    series_list = []
    # used for normalizing
    sums_datasets = [] # list which holds sum of all observations for each dataset
    dataset_durations = [] # list which holds number of time steps for each dataset
    
    # Create list to store all the CutoutDatasets
    all_datasets = []
    
    # for each subject, load data and resample, and create CutoutDataset
    for subject in subjects:
        # Finding the path for the data we want to use
        bids_path = BIDSPath(root=bids_root, subject=subject, session=session,
                             datatype=datatype,
                             extension=extension, task=task)

        # Read the data we want to use
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        # load data into memory
        print(f"Loading data for subject {subject}")
        raw.load_data()
        # resample
        print("resampling...")
        raw.resample(resample_freq)
        # get the final time step to load
        stop = int(raw.n_times * proportion_to_use)
        series = raw.get_data(picks=channel_names, start=0, stop=stop).astype(np.float32)
        # series is an array of shape (n_channels, sequence_len)

        # calculate sum of observations, later used for normalizing
        sums_datasets.append(np.sum(series, axis = 1))
        
        series_list.append(series)
        # create new instance of CutoutDataset and add to list
        dataset = CutoutDataset(series, 0, stop, time_before_cutout,
                                    time_after_cutout, cutout_duration, 
                                    subject, channel_names, resample_freq)
        all_datasets.append(dataset)
        dataset_durations.append(stop)
    
    # If standardization_info is defined, use this to standardize
    if standardization_info is not None:
        global_mean = standardization_info["mean"]
        global_std = standardization_info["std"]
        new_standardization_info = standardization_info.copy()
    else: # use the information to this dataset to normalize (used for training set)

        global_mean = sum(sums_datasets) / sum(dataset_durations)
        global_std = np.sqrt(
            sum(np.sum((series - global_mean[:, np.newaxis])**2, axis=1)
                         for series in series_list) / sum(dataset_durations)
        )
        # this is saved because we want to incorporate the info from the training set
        # to normalize the validation set
        new_standardization_info = {
            "mean": global_mean,
            "std": global_std
        }
    
    # Loop over all datasets and standardize
    for dataset in all_datasets:
        dataset.normalize(global_mean, global_std)
    
    # Concatenate all the datasets
    # also return individuals datasets
    return tud.ConcatDataset(all_datasets), all_datasets, new_standardization_info