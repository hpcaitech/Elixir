import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


class GLUEDataModule:

    task_text_field_map = {
        'cola': ['sentence'],
        'sst2': ['sentence'],
        'mrpc': ['sentence1', 'sentence2'],
        'qqp': ['question1', 'question2'],
        'stsb': ['sentence1', 'sentence2'],
        'mnli': ['premise', 'hypothesis'],
        'qnli': ['question', 'sentence'],
        'rte': ['sentence1', 'sentence2'],
        'wnli': ['sentence1', 'sentence2'],
        'ax': ['premise', 'hypothesis'],
    }

    glue_task_num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'qqp': 2,
        'stsb': 1,
        'mnli': 3,
        'qnli': 2,
        'rte': 2,
        'wnli': 2,
        'ax': 3,
    }

    loader_columns = [
        'datasets_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'start_positions',
        'end_positions',
        'labels',
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = 'mrpc',
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        self.dataset = None
        self.columns = None
        self.eval_splits = None

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset('glue', self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=['label'],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type='torch', columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]

    def train_loader_and_sampler(self):
        train_set = self.dataset['train']
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(train_set, self.train_batch_size, sampler=train_sampler)
        return train_loader, train_sampler

    def val_loader_and_sampler(self):
        valid_set = self.dataset['validation']
        valid_loader = DataLoader(valid_set, self.eval_batch_size)
        return valid_loader

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(texts_or_text_pairs,
                                                    max_length=self.max_seq_length,
                                                    pad_to_max_length=True,
                                                    truncation=True)

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['label']

        return features
