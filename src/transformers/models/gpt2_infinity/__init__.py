# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_gpt2": ["GPT2_INFINITY_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2InfinityConfig", "GPT2InfinityOnnxConfig"],
    "tokenization_gpt2": ["GPT2InfinityTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gpt2_fast"] = ["GPT2InfinityTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gpt2"] = [
        "GPT2_INFINITY_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPT2InfinityDoubleHeadsModel",
        "GPT2InfinityForSequenceClassification",
        "GPT2InfinityForTokenClassification",
        "GPT2InfinityLMHeadModel",
        "GPT2InfinityModel",
        "GPT2InfinityPreTrainedModel",
        "load_tf_weights_in_gpt2_infinity",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_gpt2"] = [
        "TF_GPT2_INFINITY_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFGPT2InfinityDoubleHeadsModel",
        "TFGPT2InfinityForSequenceClassification",
        "TFGPT2InfinityLMHeadModel",
        "TFGPT2InfinityMainLayer",
        "TFGPT2InfinityModel",
        "TFGPT2InfinityPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2InfinityLMHeadModel", "FlaxGPT2InfinityModel", "FlaxGPT2InfinityPreTrainedModel"]

if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_INFINITY_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2InfinityConfig, GPT2InfinityOnnxConfig
    from .tokenization_gpt2 import GPT2InfinityTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_gpt2_fast import GPT2InfinityTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gpt2 import (
            GPT2_INFINITY_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPT2InfinityDoubleHeadsModel,
            GPT2InfinityForSequenceClassification,
            GPT2InfinityForTokenClassification,
            GPT2InfinityLMHeadModel,
            GPT2InfinityModel,
            GPT2InfinityPreTrainedModel,
            load_tf_weights_in_gpt2_infinity,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_gpt2 import (
            TF_GPT2_INFINITY_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGPT2InfinityDoubleHeadsModel,
            TFGPT2InfinityForSequenceClassification,
            TFGPT2InfinityLMHeadModel,
            TFGPT2InfinityMainLayer,
            TFGPT2InfinityModel,
            TFGPT2InfinityPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_gpt2 import FlaxGPT2InfinityLMHeadModel, FlaxGPT2InfinityModel, FlaxGPT2InfinityPreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
