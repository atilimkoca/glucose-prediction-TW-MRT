import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, units=128):
        super(Attention, self).__init__()
        self.units = units
        # Initialize layers
        self.attention_score_vec = nn.Linear(units, units, bias=False)
        
        # Adjust this line to match the concatenated input size
        # Assuming `context_vector` and `h_t` are both (batch_size, units), 
        # their concatenation would be (batch_size, 2*units), so the input feature size should be 2*units.
        self.attention_vector = nn.Linear(2*units, units, bias=False)

    def forward(self, inputs):
        # Assuming inputs shape is (batch_size, time_steps, input_dim)
        
        # Compute the score first part
        score_first_part = self.attention_score_vec(inputs) # (batch_size, time_steps, units)

        # Get the last hidden state
        h_t = inputs[:, -1, :] # (batch_size, input_dim)

        # Compute the score
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2) # (batch_size, time_steps)
        
        # Compute the attention weights
        attention_weights = F.softmax(score, dim=-1) # (batch_size, time_steps)

        # Compute the context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1) # (batch_size, input_dim)
        
        # Concatenate context vector with last hidden state and apply non-linearity
        pre_activation = torch.cat([context_vector, h_t], dim=1) # (batch_size, 2*input_dim)
        attention_vector = torch.tanh(self.attention_vector(pre_activation)) # (batch_size, units)

        # Repeat the attention vector across the time dimension
        attention_vector_repeated = attention_vector.unsqueeze(1).repeat(1, inputs.size(1), 1) # (batch_size, time_steps, units)
        
        return attention_vector_repeated

    def compute_output_shape(self, input_shape):
        # PyTorch modules do not usually define compute_output_shape, you would just check the output shape of the forward pass
        # This is provided for completeness
        return (input_shape[0], input_shape[1], self.units)
