import torch
import torch.nn as nn



class BenchmarkModel(nn.Module):
    '''
    Model that given input and target as defined by CutoutDataset
    predicts the target section by connecting the last data point
    before the target to the first data point after the target for
    each channel and each section in a batch. 
    '''
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, _=None): #last parameter is to comply with big model
        # Get inputs
        forward_inputs = inputs[0]
        backward_inputs = inputs[1]
        # dimension is (batch_size, n_channels, sequence_length)
        
        # Get starting and ending point for prediction 
        start = forward_inputs[:, :, -1]
        end = backward_inputs[:, :, -1]
        
        # tensor for storing output
        output = torch.empty((targets[0].size()))
        
        for b in range(targets[0].size(0)): # loop over elements in batch
            for i in range(targets[0].size(1)): # loop over channels
                start_i = start[b,i].item() # starting point for specific channel and element
                end_i = end[b,i].item() # ending point
                output[b, i, :] = torch.linspace(start_i, end_i, targets[0].size(2)+2)[1:-1] # connect start end end
                
        return output