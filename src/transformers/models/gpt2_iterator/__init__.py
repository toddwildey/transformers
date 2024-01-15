# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_gpt2_iterator": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2_iterator": ["GPT2Tokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gpt2_iterator_fast"] = ["GPT2TokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gpt2_iterator"] = [
        "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPT2IteratorDoubleHeadsModel",
        "GPT2IteratorForQuestionAnswering",
        "GPT2IteratorForSequenceClassification",
        "GPT2IteratorForTokenClassification",
        "GPT2IteratorLMHeadModel",
        "GPT2IteratorModel",
        "GPT2IteratorPreTrainedModel",
        "load_tf_weights_in_gpt2",
    ]

if TYPE_CHECKING:
    from ..gpt2.configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from ..gpt2.tokenization_gpt2 import GPT2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from ..gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gpt2_iterator import (
            GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPT2IteratorDoubleHeadsModel,
            GPT2IteratorForQuestionAnswering,
            GPT2IteratorForSequenceClassification,
            GPT2IteratorForTokenClassification,
            GPT2IteratorLMHeadModel,
            GPT2IteratorModel,
            GPT2IteratorPreTrainedModel,
            load_tf_weights_in_gpt2,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
