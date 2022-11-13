#! /bin/bash .env/bin/python3

import re
from datasets import load_dataset

DATASET_NAME = "iohadrubin/wikitext-103-raw-v1"
wiki_dataset = load_dataset(DATASET_NAME)

WIKI_TITLE_REGEX = re.compile('^= (.+) =$')
def extract_wiki_title_from_dataset(example):
    example_text = example['text']
    title_line = example_text[0 : example_text.find('\n')]
    title_match = WIKI_TITLE_REGEX.match(title_line)
    title_text = title_match[1] if title_match is not None else None
    return { 'text': title_text }

wiki_titles_dataset = wiki_dataset.map(extract_wiki_title_from_dataset)

wiki_titles = []

def remove_none_from_dataset(example):
    if example['text'] is not None:
        wiki_titles.append(example['text'])

wiki_titles_dataset.map(remove_none_from_dataset)

def str_len(text):
    return len(text)

wiki_titles.sort(key = str_len)

for wiki_title in wiki_titles:
    # By inspection, anything longer than 99 characters was not a title.  The longest title was:
    # Optional Protocol to the Convention on the Elimination of All Forms of Discrimination against Women
    if len(wiki_title) < 100:
        print(wiki_title)
