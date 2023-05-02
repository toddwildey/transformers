import heapq
import json
import math
import os
import time
import torch

def partition_dataset_for_workers(examples, key_name, block_size, num_workers):
    partition_heap = []

    for idx in range(num_workers):
        heapq.heappush(partition_heap, (0, []))

    max_partition_length = 0
    min_partition = partition_heap[0]
    for example_idx, example in enumerate(examples):
        example_size = math.ceil(len(example[key_name]) / block_size)
        partition_length = min_partition[0] + example_size
        min_partition[1].append(example_idx)

        max_partition_length = max(max_partition_length, partition_length)

        heapq.heapreplace(partition_heap, (partition_length, min_partition[1]))
        min_partition = partition_heap[0]

    partitions = []
    for partition_tuple in partition_heap:
        partitions.append({
            'partition_length': partition_tuple[0],
            'example_indices': partition_tuple[1]
        })

    return {
        'block_size': block_size,
        'num_workers': num_workers,
        'max_partition_length': max_partition_length,
        'partitions': partitions
    }

def get_dataset_partition_path(root_path, num_workers, dataset_name):
    return os.path.join(root_path, str(num_workers), f'{dataset_name}.json')

def load_dataset_partitions_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.loads(file.read())
    except:
        return None

def save_dataset_partitions_to_file(file_path, dataset_partitions):
    os.makedirs(os.path.dirname(file_path), exist_ok = True)

    with open(file_path, 'w') as file:
        file.write(json.dumps(dataset_partitions))

def load_or_build_dataset_partitions(file_path, tokenized_dataset, key_name, block_size, num_workers):
    partitions = load_dataset_partitions_from_file(file_path)

    if partitions is not None:
        print(f"Loaded dataset partitions from {file_path}")
    else:
        print(f"Dataset partitions not found - partitioning the dataset")

        start_time = time.perf_counter()
        partitions = partition_dataset_for_workers(tokenized_dataset, key_name, block_size, num_workers)
        end_time = time.perf_counter()

        print(f"Partitioned the dataset in {end_time - start_time} seconds")
        save_dataset_partitions_to_file(file_path, partitions)
        print(f"Saving dataset partitions to {file_path}")

    return partitions

def group_text_for_example(example, index, key_name, block_size, pad_token):
    total_length = len(example[key_name])

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in example.items()
    }

    # Ensure everything is sized as a multiple of block_size
    for k in result.keys():
        vecs = result[k]
        if len(vecs) > 0:
            last_vec = vecs[-1]
            last_vec_len = len(last_vec)
            if last_vec_len < block_size:
                last_vec.extend([ pad_token for i in range(0, block_size - last_vec_len) ])

    result["labels"] = result["input_ids"].copy()
    result["example_idx"] = [ index for i in range(0, total_length, block_size) ]
    return result

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_sequential_texts_factory(block_size, examples_keys):
    key_name = examples_keys[0]

    def _group_sequential_texts(examples, indices):
        result = {
            k: [] for k in examples_keys
        }

        result["labels"] = []
        result["example_idx"] = []

        result_keys = result.keys()
        for idx in range(len(examples[key_name])):
            example = {
                k: examples[k][idx] for k in examples_keys
            }

            example_result = group_text_for_example(example, indices[idx], key_name, block_size, 0)

            for k in result_keys:
                result[k].extend(example_result[k])

        return result

    return _group_sequential_texts

class SequentialDatasetIterator:
    def __init__(self, dataset, partitions, block_size, max_partition_length):
        self.partition_length = partitions['partition_length']
        self.example_indices = partitions['example_indices']
        self.block_size = block_size
        self.max_partition_length = max_partition_length

        self.dataset_iterator = iter(dataset)
        self.example_idx_iterator = iter(self.example_indices)
        self.last_example_idx = next(self.example_idx_iterator)

    # TODO - handle vending till max_partition_length instead of partition_length and deletion of example_idx from returned items
    def __next__(self):
        dataset_item = next(self.dataset_iterator)
        current_example_idx = dataset_item['example_idx']

        if current_example_idx == self.last_example_idx:
            return dataset_item

        self.last_example_idx = next(self.example_idx_iterator)
        while current_example_idx != self.last_example_idx:
            dataset_item = next(self.dataset_iterator)
            current_example_idx = dataset_item['example_idx']

        return dataset_item

class SequentialIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, dataset_partitions, partition_rank):
        super(SequentialIterableDataset).__init__()
        self.dataset = dataset
        self.dataset_partitions = dataset_partitions
        self.partition_rank = partition_rank

        if partition_rank is None:
            self.dataset_partition_length = len(dataset)
        else:
            self.dataset_partition_length = dataset_partitions['max_partition_length']

    def __len__(self):
        return self.dataset_partition_length

    def __iter__(self):
        if self.partition_rank is None:
            return iter(self.dataset)

        return SequentialDatasetIterator(
            self.dataset,
            self.dataset_partitions['partitions'][self.partition_rank],
            self.dataset_partitions['block_size'],
            self.dataset_partition_length
        )
