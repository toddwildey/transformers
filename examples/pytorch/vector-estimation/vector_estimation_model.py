import torch

from dataclasses import dataclass
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.pytorch_utils import Conv1D

CROSS_ENTROPY_LOSS = nn.CrossEntropyLoss()
KL_DIV_LOSS = nn.KLDivLoss(reduction = 'batchmean', log_target = True)
MSE_LOSS = nn.MSELoss()
L1_LOSS = nn.L1Loss()

@dataclass
class VectorEstimationModelOutput:
    """
    Base class for vector estimation model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Vector estimation loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the vector estimation model (scores for each column before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class LinearVectorEstimationModel(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.iterator_1 = nn.Linear(n_embed, n_embed, bias = False)
        self.iterator_2 = nn.Linear(n_embed, n_embed, bias = False)

    def init_weights(self):
        self.iterator_1.weight.data.fill_(0.0)
        self.iterator_1.weight.data.fill_diagonal_(0.5)
        self.iterator_2.weight.data.fill_(0.0)
        self.iterator_2.weight.data.fill_diagonal_(0.5)

    def forward(
        self,
        input_feature_vectors: Optional[torch.Tensor] = None,
        target_feature_vector: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[VectorEstimationModelOutput, None]:
        if input_feature_vectors.nelement() == 0 or input_feature_vectors.size(1) < 2:
            return None

        iterator_outputs_1 = self.iterator_1(torch.select(input_feature_vectors, 1, 0))
        iterator_outputs_2 = self.iterator_2(torch.select(input_feature_vectors, 1, 1))

        iterator_outputs = iterator_outputs_1 + iterator_outputs_2

        # We rely on accelerate to move target_feature_vector to the correct device beforehand
        loss = CROSS_ENTROPY_LOSS(iterator_outputs, target_feature_vector)

        # print('loss')
        # print(loss)

        return VectorEstimationModelOutput(
            loss = loss,
            logits = iterator_outputs,
        )

class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop = 0.1):
        super().__init__()
        self.c_fc = Conv1D(n_embed, n_embed)
        self.c_proj = Conv1D(n_embed, n_embed)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class AttentionVectorEstimationModel(nn.Module):
    def __init__(self, n_embed, layer_norm_epsilon = 0.00001):
        super().__init__()
        self.n_embed = n_embed
        self.c_attn = Conv1D(3 * n_embed, n_embed)
        self.ln_1 = nn.LayerNorm(n_embed, eps = layer_norm_epsilon)
        self.attention = nn.MultiheadAttention(n_embed, 8)
        self.ln_2 = nn.LayerNorm(n_embed, eps = layer_norm_epsilon)
        self.mlp = MLP(n_embed)

    def forward(self, input_feature_vectors: Optional[torch.FloatTensor]) -> Tuple[torch.FloatTensor]:
        residual = input_feature_vectors
        hidden_states = self.ln_1(input_feature_vectors)
        query, key, value = self.c_attn(hidden_states).split(self.n_embed, dim = 2)
        attn_output, attn_output_weights = self.attention(query, key, value)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

# AttentionVectorEstimationModel does not work - do not use it
class AttentionVectorEstimationModel(nn.Module):
    def __init__(self, n_embed, layer_norm_epsilon = 0.00001):
        super().__init__()
        self.n_embed = n_embed
        self.c_attn = Conv1D(3 * n_embed, n_embed)
        self.ln_1 = nn.LayerNorm(n_embed, eps = layer_norm_epsilon)
        self.attention = nn.MultiheadAttention(n_embed, 8)
        self.ln_2 = nn.LayerNorm(n_embed, eps = layer_norm_epsilon)
        self.mlp = MLP(n_embed)

    def init_weights(self):
        pass

    def forward(
        self,
        input_feature_vectors: Optional[torch.Tensor] = None,
        target_feature_vector: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[VectorEstimationModelOutput, None]:
        if input_feature_vectors.nelement() == 0 or input_feature_vectors.size(1) < 2:
            return None

        residual = input_feature_vectors
        hidden_states = self.ln_1(input_feature_vectors)
        query, key, value = self.c_attn(hidden_states).split(self.n_embed, dim = 2)
        attn_output, attn_output_weights = self.attention(query, key, value)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        print('hidden_states')
        print(hidden_states)
        print('hidden_states.size()')
        print(hidden_states.size())

        # expected_state = torch.select(input_feature_vectors, 1, 0) + torch.select(input_feature_vectors, 1, 1)
        expected_state = torch.sum(hidden_states, dim = 1)

        # We rely on accelerate to move target_feature_vector to the correct device beforehand
        loss = CROSS_ENTROPY_LOSS(expected_state, target_feature_vector)

        print('loss')
        print(loss)

        return VectorEstimationModelOutput(
            loss = loss,
            logits = expected_state,
        )

# Loss doesn't go down meaningfully for this model
class LSTMVectorEstimationModel(nn.Module):
    def __init__(self, n_embed, num_layers = 2, device = None):
        super().__init__()
        self.n_embed = n_embed
        self.attention = [ self.create_lstm(n_embed, device) for idx in range(0, num_layers) ]

    def create_lstm(self, n_embed, device):
        return nn.LSTM(
            n_embed,
            n_embed,
            device = device,
            bias = False,
            batch_first = True
        )

    def init_weights(self):
        pass

    def forward(
        self,
        input_feature_vectors: Optional[torch.Tensor] = None,
        target_feature_vector: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[VectorEstimationModelOutput, None]:
        if input_feature_vectors.nelement() == 0 or input_feature_vectors.size(1) < 2:
            return None

        output = None
        # hidden_states = None

        # Testing using the input vector as h_0
        h_0 = torch.zeros(1, input_feature_vectors.size(0), input_feature_vectors.size(2), device = input_feature_vectors.device)
        c_0 = input_feature_vectors[:, 1:2, :].transpose(0, 1).contiguous()
        # h_0 = input_feature_vectors[:, 0:1, :].transpose(0, 1).contiguous()
        # c_0 = torch.zeros(1, input_feature_vectors.size(0), input_feature_vectors.size(2), device = input_feature_vectors.device)
        hidden_states = ( h_0, c_0 )
        for idx in range(0, input_feature_vectors.size(1) - 1):
            input_vector = input_feature_vectors[:, idx:idx+1, :]

            # print('input_vector')
            # print(input_vector)
            # print('input_vector.size()')
            # print(input_vector.size())

            output, hidden_states = self.attention[idx](input_vector, hidden_states)

            # print('output')
            # print(output)
            # print('output.size()')
            # print(output.size())

        # # Testing using the input vector as h_0
        # h_0 = input_feature_vectors[:, 0:1, :].transpose(0, 1).contiguous()
        # c_0 = torch.zeros(1, input_feature_vectors.size(0), input_feature_vectors.size(2), device = input_feature_vectors.device)
        # hidden_states = ( h_0, c_0 )
        # for idx in range(1, input_feature_vectors.size(1)):
        #     input_vector = input_feature_vectors[:, idx:idx+1, :]

        #     # print('input_vector')
        #     # print(input_vector)
        #     # print('input_vector.size()')
        #     # print(input_vector.size())

        #     output, hidden_states = self.attention[idx](input_vector, hidden_states)

        #     # print('output')
        #     # print(output)
        #     # print('output.size()')
        #     # print(output.size())

        # print('target_feature_vector')
        # print(target_feature_vector)
        # print('target_feature_vector.size()')
        # print(target_feature_vector.size())

        # We rely on accelerate to move target_feature_vector to the correct device beforehand
        # loss = CROSS_ENTROPY_LOSS(output[:, 0, :], target_feature_vector)

        # Results in NaN
        # ln_output = torch.log(output[:, 0, :])
        # loss = KL_DIV_LOSS(ln_output, target_feature_vector)

        # loss = MSE_LOSS(output[:, 0, :], target_feature_vector)
        loss = L1_LOSS(output[:, 0, :], target_feature_vector)

        # print('loss')
        # print(loss)

        return VectorEstimationModelOutput(
            loss = loss,
            logits = hidden_states,
        )

# This model does not fit on 4090 GTX
class InfoMetricVectorEstimationModel(nn.Module):
    def __init__(self, n_embed, device = None, dtype = None) -> None:
        factory_kwargs = { 'device': device, 'dtype': dtype }
        super().__init__()
        self.n_embed = n_embed
        self.weight = nn.parameter.Parameter(torch.empty((n_embed, n_embed, n_embed), **factory_kwargs))

    def init_weights(self) -> None:
        pass

    def forward(
        self,
        input_feature_vectors: Optional[torch.Tensor] = None,
        target_feature_vector: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[VectorEstimationModelOutput, None]:
        if input_feature_vectors.nelement() == 0 or input_feature_vectors.size(1) < 2:
            return None

        outer_product = torch.bmm(
            input_feature_vectors[:, 0:1, :].transpose(1, 2),
            input_feature_vectors[:, 1:2, :]
        )

        outer_product.requires_grad = False

        # print('outer_product')
        # print(outer_product)
        # print('outer_product.size()')
        # print(outer_product.size())

        hidden_states = torch.einsum('bij,oij->bo', outer_product, self.weight)

        # print('hidden_states')
        # print(hidden_states)
        # print('hidden_states.size()')
        # print(hidden_states.size())

        loss = L1_LOSS(hidden_states, target_feature_vector)

        print('loss')
        print(loss)

        return VectorEstimationModelOutput(
            loss = loss,
            logits = hidden_states,
        )
