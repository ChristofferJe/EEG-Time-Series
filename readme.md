# Time Series Prediction and Anomaly Detection in EEG Data
The repository contains the code used in the project described in `report.pdf`. The project attempts to use a encoder-decoder to predict and detect anomalies in multivariate EGG time series data. Both the encoder and decoder consists of a multi-layer LSTM. The models are implemented in Python using PyTorch. Below is a very short description of the different files in the repository.

### Notebook files:

- `hyperparameter_serach.ipynb`: Notebook used for searching for hyperparameters

- `cross_validation.ipynb`: Notebook used for evaluating the model by cross-validation

- `benchmark_run.ipynb`: Testing out the benchmark model to compare with the results from cross validation

- `anomaly_detection.ipynb`: Trying to use the benchmark model and the seq2seq model for anomaly detection

### .py files:

- `two_way_seq2seq.py`: Where the predictive model is defined

- `bidirectional_dataset.py`: The dataset used for training and evaluating the predictive model

- `anomaly_dataset.py`: The dataset where synthetic anomalies are inserted into sections

- `trainer.py`: The training loop used

- `plotting.py`: Functions used for making nice plots

