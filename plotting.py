import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

# list of possible colors to use
COLORS = tuple(mcolors.TABLEAU_COLORS.keys())

def timeseries_plot(series, channels : list, start_steps : int, linewidth = 0.3,
                    ymin = None, ymax = None, sfreq=20, bg_color=(1.0,1.0,1.0)):
    '''Plots the data. Should have shape (n_channels, length)
    The start_steps is simply used for making the axis have the proper
    time values
    Args:
        data: array where each row is a channel
        channels: list of channel names (used for plot legend)
        start_steps: the first time step of the data (is converted to seconds
            and used for the axis)
        linewidth: Width of line
        ymin: lowest y value to include (if not specified, matplotlib decides)
        ymax: largets y value to include
        sfreq: the sampling frequence (used for converting time steps to seconds)
        bg_color: tuple (r, g, b) which specifies the background color to use
            (default is white)
    '''
    length = series.shape[1]
    # create "second vector" to plot against
    sec = np.linspace(start_steps / sfreq, (start_steps + length) / sfreq, length)
    
    # plot each channel with a differnt color
    fig, ax = plt.subplots()
    for i, channel in enumerate(channels):
      ax.plot(sec, series[i], linewidth = linewidth, color = COLORS[i], label=channel)

    if ymin is not None:
      ax.set_ylim(ymin, ymax)
    
    ax.set_facecolor(bg_color)
    
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.show()


def plot_target(datasets, idx, dataset_idx, linewidth = 0.3):
    '''Plot the stuff before the cutout, the cutout and the stuff after the cutout
    for the section with index idx in CutoutDataset with index dataset_index in
    the list datasets
    Args:
        datasets: List of CutoutDatasets
        idx: The index of the section to plot
        dataset_idx: The index of the CutoutDataset to choose the section from
        linewidth: Width of line
    '''
    # get the correct CutoutDataset
    dataset = datasets[dataset_idx]
    channel_names = dataset.get_channel_names()
    # get the section to plot
    (forward_input, backward_input), (forward_target, backward_target) = dataset[idx]

    # flip the backward_input so that we plot it in the right direction
    input_after = np.flip(backward_input, axis = 1) 

    # Concate the input with the target
    cat_input_target = np.concatenate((forward_input, forward_target, input_after), axis=1)

    timeseries_plot(cat_input_target, channel_names, 0, linewidth=linewidth)


def plot_anomaly_target(datasets, idx, dataset_idx, linewidth = 0.3):
    '''Plot the stuff before the cutout, the cutout and the stuff after the cutout
    for the section with index idx in AnomalyDataset with index dataset_index in
    the list datasets. If there is an anomaly, a red background color is used
    Args:
        datasets: List of CutoutDatasets
        idx: The index of the section to plot
        dataset_idx: The index of the CutoutDataset to choose the section from
        linewidth: Width of line
    '''
    dataset = datasets[dataset_idx]
    channel_names = dataset.get_channel_names()
    (forward_input, backward_input), (forward_target, backward_target), has_anomaly = dataset[idx]

    # flip the backward input in time to swap it around
    input_after = np.flip(backward_input, axis = 1) 

    # Concate the input with the target
    cat_input_target = np.concatenate((forward_input, forward_target, input_after), axis=1)

    color = (1.0, 0.8, 0.8) if has_anomaly else (1.0, 1.0, 1.0) #red background if anomaly
    timeseries_plot(cat_input_target, channel_names, 0, bg_color = color,
                     linewidth=linewidth)



def plot_target_and_prediction(model, datasets, idx, dataset_idx, device, linewidth = 0.3,
                              time_before_cutout=1, cutout_duration=1, time_after_cutout=1):
    '''Plot the stuff before the cutout, the cutout and the stuff after the cutout
    for the section with index idx in CutoutDataset with index dataset_index in
    the list datasets. The specified model is used to predict the cutout, and the
    predictions are drawn with a dashed line
    Args:
        model: A Pytorch model to get the predictions from
        datasets: List of CutoutDatasets
        idx: The index of the section to plot
        dataset_idx: The index of the CutoutDataset to choose the section from
        device: Device (GPU/CPU) to perform the predictions on
        linewidth: Width of line
        time_before_cutout: Time in seconds to include before the cutout in each example
            cutout_duration: Time in seconds of the cutout duration
        time_after_cutout: Time in seconds to include after the cutout
    '''

    dataset = datasets[dataset_idx]
    channel_names = dataset.get_channel_names()
    (forward_input, backward_input), (forward_target, backward_target) = dataset[idx]

    # flip the backward input in time to swap it around
    input_after = np.flip(backward_input, axis = 1) 

    # Concate the input with the target
    cat_input_target = np.concatenate((forward_input, forward_target, input_after), axis=1)

    # x-axis for concatenated target/input showing time in seconds
    length_cat = cat_input_target.shape[1]
    x_cat = np.linspace(0, (time_before_cutout+cutout_duration+time_after_cutout), length_cat)

    # Get the output of the model
    model.eval()
    inputs = (torch.tensor(forward_input).unsqueeze(0).to(device), torch.tensor(backward_input).unsqueeze(0).to(device))
    targets = (torch.tensor(forward_target).unsqueeze(0).to(device), torch.tensor(backward_target).unsqueeze(0).to(device))
    with torch.no_grad():
       output = model(inputs, targets, 0).squeeze().to("cpu")

    # x-axis for the output of the model
    x_output = np.linspace(time_before_cutout, time_before_cutout+cutout_duration, output.shape[1])


    # plot each channel and its prediction
    fig, ax = plt.subplots()
    for i, channel in enumerate(channel_names):
        ax.plot(x_cat, cat_input_target[i], linewidth = linewidth, color = COLORS[i], label=channel)
        ax.plot(x_output, output[i], linewidth = linewidth*3, color = COLORS[i], linestyle="dashed")
        
    plt.legend()
    plt.show()
    return fig





