from accelerate import Accelerator
import torch
import datasets
from datasets import load_dataset

from transformers import \
    AutoTokenizer

from sequential_iterable_dataset import \
    get_dataset_partition_path, \
    load_or_build_dataset_partitions, \
    load_dataset_partitions_from_file, \
    group_sequential_texts_factory, \
    SequentialIterableDataset

accelerator = Accelerator()

model_name_or_path = "gpt2"
dataset_name = "pg19"
dataset_config_name = "pg19"
dataset_map_num_proc = 1
MODEL_BLOCK_SIZE = 512

dataset_partitions_path = 'infinite_memory_transformer_sticky_mem/partitions'

num_processes = accelerator.num_processes

raw_datasets = load_dataset(dataset_name, dataset_config_name)
column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_datasets = raw_datasets['train'].map(
    tokenize_function,
    batched = True,
    num_proc = dataset_map_num_proc,
    load_from_cache_file = True,
    desc = "Running tokenizer on dataset",
)

dataset_partitions_path = get_dataset_partition_path(
        dataset_partitions_path,
        num_processes,
        dataset_name
)

if accelerator.is_main_process:
    dataset_partitions = load_or_build_dataset_partitions(
        dataset_partitions_path,
        tokenized_datasets,
        'input_ids',
        MODEL_BLOCK_SIZE,
        num_processes
    )

accelerator.wait_for_everyone()

if not accelerator.is_main_process:
    dataset_partitions = load_dataset_partitions_from_file(dataset_partitions_path)

# print('dataset_partitions')
# print(dataset_partitions)

EXAMPLE_KEYS = [ "input_ids", "attention_mask" ]

grouped_datasets = tokenized_datasets.map(
    group_sequential_texts_factory(MODEL_BLOCK_SIZE, EXAMPLE_KEYS),
    batched = True,
    with_indices = True,
    num_proc = dataset_map_num_proc,
    load_from_cache_file = True,
    remove_columns = column_names,
    desc = f"Grouping texts into chunks",
)

print(f'Assuming dataset partition #{accelerator.process_index}')

dataset = SequentialIterableDataset(
        grouped_datasets,
        dataset_partitions,
        accelerator.process_index
)

dataset_partition = dataset_partitions['partitions'][accelerator.process_index]
dataset_partition_example_indices = set(dataset_partition['example_indices'])

# Single-process loading
dataloader = torch.utils.data.DataLoader(dataset, num_workers = 0)

print(f'Dataloader length: {len(dataloader)}')

for idx, data in enumerate(dataloader):
    # print("len(data['input_ids'])")
    # print(len(data['input_ids']))
    # print("len(data['attention_mask'])")
    # print(len(data['attention_mask']))
    # print("len(data['labels'])")
    # print(len(data['labels']))
    # print("len(data['example_idx'])")
    # print(len(data['example_idx']))
    # print("data['example_idx']")
    # print(data['example_idx'])

    print(f'Validating batch #{idx}')

    assert(len(data['input_ids']) == MODEL_BLOCK_SIZE)
    assert(len(data['attention_mask']) == MODEL_BLOCK_SIZE)
    assert(len(data['labels']) == MODEL_BLOCK_SIZE)
    assert(len(data['example_idx']) == 1)
    assert(isinstance(data['example_idx'].item(), int))
    assert(data['example_idx'].item() in dataset_partition_example_indices)

    print(f'Validated batch #{idx}')

    # accelerator.wait_for_everyone()
