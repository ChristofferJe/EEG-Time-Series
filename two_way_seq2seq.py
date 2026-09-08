import torch
import torch.nn as nn
import random

## Based on https://medium.com/@maxbrenner-ai/implementing-seq2seq-models-for-efficient-time-series-forecasting-88dba1d66187

class Encoder(nn.Module):
    '''Ecoder used in the two way seq2seq model'''
    def __init__(self, hidden_size, n_channels = 7, num_layers = 2, dropout = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_layers = num_layers, 
                          input_size = 7, 
                          hidden_size=hidden_size, 
                          batch_first=True, 
                          bidirectional=False, 
                          dropout=dropout)

    def forward(self, encoder_inputs):
        outputs, (hidden, cell) = self.lstm(encoder_inputs)
        # Only use hidden from last layer
        return outputs, (hidden[-1, :, :].unsqueeze(0), cell[-1, :, :].unsqueeze(0))
    
    
class Decoder(nn.Module):
    '''Decoder used in the two way seq2seq model'''
    def __init__(self, hidden_size, n_channels=7):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, batch_first = True, num_layers = 1)
        self.out = nn.Linear(hidden_size, n_channels)

    def forward(self, initial_input, hidden, cell, targets,
                teacher_force_probability):
        
        # Get the lenght of the target sequence
        decoder_sequence_length = targets.shape[1]
        
        # Empty list to save the outputs for each timepoint t
        outputs = []

        # Input given to the decoder at time t, initaly the last encoder input
        input_at_t = initial_input[:, -1, :].unsqueeze(1) 

        # Decoder is used to predict the value for each timepoint t
        for t in range(decoder_sequence_length):
            output, (hidden, cell) = self.lstm(input_at_t, (hidden, cell))
            outputs.append(self.out(output))

            # Teacherforcing is used at given probability
            teacher_force = random.random() < teacher_force_probability

            # Determing wheter teatcher forcing or current output is given as input at next timepoint
            input_at_t = targets[:,t,:].unsqueeze(1) if teacher_force else outputs[t]

        return torch.cat(outputs, dim = 1)
    
    
class TwoWaySeq2Seq(nn.Module):
    '''A sequence model consisting of two encoder and decoder pairs using RNN's in form of LSTM. 
    There is an encoder and decoder pair moving both forward and backward over the input.
    The output of the two decoders are combined using a linear layer to calculate one
    output for the given input.'''
    def __init__(self, hidden_size, n_channels, cutout_duration,
                 encoder_dropout, num_layers):
        """
        Args:
            hidden_size: the size of the hidden vector in the LSTM used in the encoders and decoders.
            n_channels: the dimension of the input data
            cutout_duration: the length of the target in seconds
            encoder_dropout: the probability of dropout used in LSTM in the encoders
            num_layers: the number of layers of LSTM to use in the encoders
        """
        super().__init__()
        self.encoder_forward = Encoder(hidden_size, dropout=encoder_dropout, num_layers=num_layers)
        self.encoder_backward = Encoder(hidden_size, dropout=encoder_dropout, num_layers=num_layers)
        self.decoder_forward = Decoder(hidden_size)
        self.decoder_backward = Decoder(hidden_size)
        self.cutout_duration = cutout_duration
        self.out = nn.Linear(2*n_channels, n_channels)


    def forward(self, encoder_inputs, targets, teacher_force_probability):
        # Split input up in the two directions
        encoder_forward_inputs = encoder_inputs[0].permute((0, 2, 1)) # shape: (batch_size, seq_len, n_chan)
        encoder_backward_inputs = encoder_inputs[1].permute((0, 2, 1)) # shape: (batch_size, seq_len, n_chan)

        # Split targets up in the two directions
        targets_forward = targets[0].permute((0, 2, 1)) # shape: (batch_size, seq_len, n_chan)
        targets_backward = targets[1].permute((0, 2, 1)) # shape: (batch_size, seq_len, n_chan)

        # Use encoder in both directions
        _, (hidden_forward, cell_forward) = self.encoder_forward(encoder_forward_inputs)
        _, (hidden_backward, cell_backward) = self.encoder_backward(encoder_backward_inputs) 
        # Encoder outputs shape: (batch_size, seq_len, hidden_size)
        # Hidden shape: (1, batch_size, hidden_size)
        # Cell shape: (1, batch_size, hidden_size)

        # Use decoder in both directions, the outputs have shape: batch_size, sequence_length, n_channels 
        outputs_forward = self.decoder_forward(encoder_forward_inputs, 
                                               hidden_forward, 
                                               cell_forward, 
                                               targets_forward, 
                                               teacher_force_probability)
        outputs_backward = self.decoder_backward(encoder_backward_inputs, 
                                                 hidden_backward, 
                                                 cell_backward, 
                                                 targets_backward, 
                                                 teacher_force_probability)
        
        # Flip outputs_backwards, so it macthes with forward
        outputs_backward =  torch.flip(outputs_backward, dims=[1]) # shape: batch_size, sequence_length, n_channels

        # Combine the output for both directions into one output
        outputs = torch.cat((outputs_forward, outputs_backward), dim=2) 
        outputs = self.out(outputs)

        return outputs.permute((0, 2, 1))