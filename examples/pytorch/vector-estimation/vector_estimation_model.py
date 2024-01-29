import torch

from dataclasses import dataclass
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

CROSS_ENTROPY_LOSS = nn.CrossEntropyLoss()

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
    def __init__(self, n_embd):
        super().__init__()
        self.iterator_1 = nn.Linear(n_embd, n_embd, bias = False)
        self.iterator_2 = nn.Linear(n_embd, n_embd, bias = False)

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
        if input_feature_vectors.nelement() == 0:
            return None

        iterator_outputs_1 = self.iterator_1(torch.select(input_feature_vectors, 1, 0))
        iterator_outputs_2 = self.iterator_2(torch.select(input_feature_vectors, 1, 1))

        iterator_outputs = iterator_outputs_1 + iterator_outputs_2

        # We rely on accelerate to move target_feature_vector to the correct device beforehand
        loss = CROSS_ENTROPY_LOSS(iterator_outputs, target_feature_vector)

        return VectorEstimationModelOutput(
            loss = loss,
            logits = iterator_outputs,
        )
