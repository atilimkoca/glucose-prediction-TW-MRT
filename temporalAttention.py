import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, units=128):
        super(TemporalAttention, self).__init__()
        self.units = units
        self.attention_score_vec = nn.Linear(units, units, bias=False)
        self.attention_vector = nn.Linear(2 * units, units, bias=False)

    def forward(self, inputs):
        score_first_part = self.attention_score_vec(inputs)  # (batch_size, time_steps, units)
        h_t = inputs[:, -1, :]  # (batch_size, input_dim)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)  # (batch_size, time_steps)
        attention_weights = F.softmax(score, dim=-1)  # (batch_size, time_steps)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)  # (batch_size, input_dim)
        pre_activation = torch.cat([context_vector, h_t], dim=1)  # (batch_size, 2*input_dim)
        attention_vector = torch.tanh(self.attention_vector(pre_activation))  # (batch_size, units)
        attention_vector_repeated = attention_vector.unsqueeze(1).repeat(1, inputs.size(1), 1)  # (batch_size, time_steps, units)
        return attention_vector_repeated
