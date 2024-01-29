<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Vector estimation training

Training models to combine feature vectors for specific language models into summary feature vectors using a a text dataset.

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

### Estimating summary feature vectors for GPT-2

The following example trains a model to estimate the feature vector output by GPT-2 for a batch of text using feature vectors output by GPT-2 for subsections of the same text.

To run on your own training and validation files, use the following command:

```bash
source .env/bin/activate

python examples/pytorch/vector-estimation/run_ve_no_trainer.py \
    --dataset_name "../data/gpt2-large/wikimedia/wikipedia/20231101.en" \
    --output_dir "../models/gpt2-large_iterator-linear/" \
    --resume_from_checkpoint "../models/gpt2-large_iterator-linear/" \
    --checkpointing_steps 100000 \
    2>&1 | tee output
```
