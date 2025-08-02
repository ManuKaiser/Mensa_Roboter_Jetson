import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):
        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        look_up = args[5]

        numNodes = len(look_up)
        model_device = next(self.parameters()).device  # Automatically detect model's device

        # Ensure hidden and cell states are on the model's device
        hidden_states = hidden_states.to(model_device)
        if cell_states is not None:
            cell_states = cell_states.to(model_device)

        outputs = torch.zeros(self.seq_length * numNodes, self.output_size, device=model_device)

        # For each frame in the sequence
        for framenum, frame in enumerate(input_data):
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            if len(nodeIDs) == 0:
                continue

            list_of_nodes = [look_up[x] for x in nodeIDs if x in look_up]
            corr_index = torch.LongTensor(list_of_nodes).to(model_device)

            nodes_current = frame[list_of_nodes, :].to(model_device)
            grid_current = grids[framenum].to(model_device)

            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            social_tensor = social_tensor.to(model_device)

            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, hidden_states_current)

            outputs.index_copy_(0, framenum * numNodes + corr_index, self.output_layer(h_nodes))
            hidden_states.index_copy_(0, corr_index, h_nodes)
            if not self.gru:
                cell_states.index_copy_(0, corr_index, c_nodes)

        outputs_return = torch.zeros(self.seq_length, numNodes, self.output_size, device=model_device)
        for framenum in range(self.seq_length):
            outputs_return[framenum, :, :] = outputs[framenum * numNodes : (framenum + 1) * numNodes, :]

        return outputs_return, hidden_states, cell_states


