from transformers import HfArgumentParser

from prompteval import PromptEvalConfig
import dataclasses
import json
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW

from tqdm import tqdm
import numpy as np

import torch.nn as nn
from torch.nn import CrossEntropyLoss

from data import load_data

from collections import defaultdict, namedtuple
from typing import *

import numpy as np
from torch.utils.data import Dataset
from openprompt.utils.logging import logger
from typing import Union
from trainer import PromptTrainer


class FewShotSampler(object):
    '''
    Few-shot learning is an important scenario for prompt-learning, this is a sampler that samples few examples over each class.

    Args:
        num_examples_total(:obj:`int`, optional): Sampling strategy ``I``: Use total number of examples for few-shot sampling.
        num_examples_per_label(:obj:`int`, optional): Sampling strategy ``II``: Use the number of examples for each label for few-shot sampling.
        also_sample_dev(:obj:`bool`, optional): Whether to apply the sampler to the dev data.
        num_examples_total_dev(:obj:`int`, optional): Sampling strategy ``I``: Use total number of examples for few-shot sampling.
        num_examples_per_label_dev(:obj:`int`, optional): Sampling strategy ``II``: Use the number of examples for each label for few-shot sampling.

    '''

    def __init__(self,
                 num_examples_total: Optional[int]=None,
                 num_examples_per_label: Optional[int]=None,
                 also_sample_dev: Optional[bool]=False,
                 num_examples_total_dev: Optional[int]=None,
                 num_examples_per_label_dev: Optional[int]=None,
                 ):
        if num_examples_total is None and num_examples_per_label is None:
            raise ValueError("num_examples_total and num_examples_per_label can't be both None.")
        elif num_examples_total is not None and num_examples_per_label is not None:
            raise ValueError("num_examples_total and num_examples_per_label can't be both set.")

        if also_sample_dev:
            if num_examples_total_dev is not None and num_examples_per_label_dev is not None:
                raise ValueError("num_examples_total and num_examples_per_label can't be both set.")
            elif num_examples_total_dev is None and num_examples_per_label_dev is None:
                logger.warning(r"specify neither num_examples_total_dev nor num_examples_per_label_dev,\
                                set to default (equal to train set setting).")
                self.num_examples_total_dev = num_examples_total
                self.num_examples_per_label_dev = num_examples_per_label
            else:
                self.num_examples_total_dev  = num_examples_total_dev
                self.num_examples_per_label_dev = num_examples_per_label_dev

        self.num_examples_total = num_examples_total
        self.num_examples_per_label = num_examples_per_label
        self.also_sample_dev = also_sample_dev

    def __call__(self,
                 train_dataset: Union[Dataset, List],
                 valid_dataset: Optional[Union[Dataset, List]] = None,
                 seed: Optional[int] = None
                ) -> Union[Dataset, List]:
        '''
        The ``__call__`` function of the few-shot sampler.

        Args:
            train_dataset (:obj:`Union[Dataset, List]`): The train datset for the sampler.
            valid_dataset (:obj:`Union[Dataset, List]`, optional): The valid datset for the sampler. Default to None.
            seed (:obj:`int`, optional): The random seed for the sampling.

        Returns:
            :obj:`(Union[Dataset, List], Union[Dataset, List])`: The sampled dataset (train_dataset, valid_dataset), whose type is identical to the input.

        '''
        if valid_dataset is None:
            if self.also_sample_dev:
                return self._sample(train_dataset, seed, sample_twice=True)
            else:
                return self._sample(train_dataset, seed, sample_twice=False)

    def _sample(self,
                data: Union[Dataset, List],
                seed: Optional[int],
                sample_twice = False,
               ) -> Union[Dataset, List]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        indices = [i for i in range(len(data))]

        if self.num_examples_per_label is not None:
            labels = [x.label for x in data]
            selected_ids = self.sample_per_label(indices, labels, self.num_examples_per_label) # TODO fix: use num_examples_per_label_dev for dev

        if sample_twice:
            selected_set = set(selected_ids)
            remain_ids = [i for i in range(len(data)) if i not in selected_set]
            if self.num_examples_per_label_dev is not None:
                remain_labels = [x.label for idx, x in enumerate(data) if idx not in selected_set]
                selected_ids_dev = self.sample_per_label(remain_ids, remain_labels, self.num_examples_per_label_dev)
            else:
                selected_ids_dev = self.sample_total(remain_ids, self.num_examples_total_dev)

            return [data[i] for i in selected_ids], [data[i] for i in selected_ids_dev]

        else:
            return [data[i] for i in selected_ids]


    def sample_total(self, indices: List, num_examples_total):
        '''
        Use the total number of examples for few-shot sampling (Strategy ``I``).

        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            num_examples_total(:obj:`int`): The total number of examples.

        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.

        '''
        self.rng.shuffle(indices)
        selected_ids = indices[:num_examples_total]
        logger.info("Selected examples (mixed) {}".format(selected_ids))
        return selected_ids

    def sample_per_label(self, indices: List, labels, num_examples_per_label):
        '''
        Use the number of examples per class for few-shot sampling (Strategy ``II``).
        If the number of examples is not enough, a warning will pop up.

        Args:
            indices(:obj:`List`): The random indices of the whole datasets.
            labels(:obj:`List`): The list of the labels.
            num_examples_per_label(:obj:`int`): The total number of examples for each class.

        Returns:
            :obj:`List`: The selected indices with the size of ``num_examples_total``.
        '''

        ids_per_label = defaultdict(list)
        selected_ids = []
        for idx, label in zip(indices, labels):
            ids_per_label[label].append(idx)
        for label, ids in ids_per_label.items():
            tmp = np.array(ids)
            self.rng.shuffle(tmp)
            if len(tmp) < num_examples_per_label:
                logger.info("Not enough examples of label {} can be sampled".format(label))
            selected_ids.extend(tmp[:num_examples_per_label].tolist())
        selected_ids = np.array(selected_ids)
        self.rng.shuffle(selected_ids)
        selected_ids = selected_ids.tolist()
        logger.info("Selected examples {}".format(selected_ids))
        return selected_ids

@dataclasses.dataclass
class FineTuneConfig(PromptEvalConfig):
    method: str = 'fine_tune'

def convert_examples_to_features(examples, max_length, tokenizer):
    features = []

    for example in examples:
        # TODO text_b
        inputs = tokenizer(example.text_a, padding="max_length", max_length=max_length, truncation=True)
        features.append({
            **inputs,
            'labels': example.label
        })
    return features

task_classes = {
    'sst2': ['negative', 'positive']
}

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "result.json")
    json_string = json.dumps(results, indent=2, sort_keys=True) + "\n"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_string)


from sklearn.metrics import f1_score, accuracy_score
import os
import json


from transformers.utils import logging
from transformers.trainer_utils import set_seed
from transformers.utils import logging
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np

logger = logging.get_logger(__name__)

class FineTuneTrainer(PromptTrainer):
    def __init__(self, model, train_dataloader, dev_dataloader, config, device, input_keys):
        super().__init__(model, train_dataloader, dev_dataloader, config, device)
        self.input_keys = input_keys

    def evaluate(self, eval_dataloader=None):
        if eval_dataloader is None:
            eval_dataloader = self.dev_dataloader

        self.model.eval()
        all_preds = []
        all_labels = []

        for step, inputs in enumerate(eval_dataloader):

            convert_inputs = {}

            for key, input in zip(self.input_keys, inputs):
                convert_inputs[key] = input.to(self.device)

            outputs = self.model(**convert_inputs)
            logits = outputs.logits
            labels = convert_inputs['labels']

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        metrics = {"eval_acc": acc, "eval_f1": f1} 

        if self.is_training:
            self.log(metrics)
        
        return metrics

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        model = self.model

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.lr)
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps) 


    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()

        convert_inputs = {}

        for key, input in zip(self.input_keys, inputs):
            convert_inputs[key] = input.to(self.device)

        outputs = model(**convert_inputs)
        loss = outputs[0]

        loss.backward()

        return loss.detach()


def main():
    parser = HfArgumentParser(FineTuneConfig)
    config, = parser.parse_args_into_dataclasses()
    config.save_to_json()

    set_seed(config.seed)

    model_config = AutoConfig.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model_config.num_labels=len(task_classes[config.task])
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        config=model_config)

    # papre data
    dataset = load_data(config.task, splits=['train', 'validation'])

    sampler = FewShotSampler(num_examples_per_label=config.shot, also_sample_dev=True)
    dataset['train'], dataset['dev'] = sampler(dataset['train'], seed=config.seed)
    
    train_features = convert_examples_to_features(dataset['train'], config.max_seq_length, tokenizer)
    dev_features = convert_examples_to_features(dataset['dev'], config.max_seq_length, tokenizer)
    test_features = convert_examples_to_features(dataset['validation'], config.max_seq_length, tokenizer)

    input_keys = []
    train_tensor_features = []
    dev_tensor_features = []
    test_tensor_features = []
    for key in train_features[0].keys():
        input_keys.append(key)
        train_tensor_features.append(torch.tensor([f[key] for f in train_features], dtype=torch.long))
        dev_tensor_features.append(torch.tensor([f[key] for f in dev_features], dtype=torch.long))
        test_tensor_features.append(torch.tensor([f[key] for f in test_features], dtype=torch.long))

    train_data = TensorDataset(*train_tensor_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    dev_data = TensorDataset(*dev_tensor_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config.eval_batch_size)

    test_data = TensorDataset(*test_tensor_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.eval_batch_size)

    trainer = FineTuneTrainer(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        config=config,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        input_keys=input_keys
    )

    trainer.train()
    logger.info("Test Results:")
    results = trainer.evaluate(test_dataloader)
    logger.info(results)

    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()

