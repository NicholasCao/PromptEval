'''

Generic prompt evaluation scripts wrapper

'''
import os
import dataclasses
import json
import torch
import logging
from typing import *

from openprompt import PromptDataLoader, Template, Verbalizer, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate, PtuningTemplate, ManualVerbalizer, SoftVerbalizer
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed

from data import load_data
from trainer import PromptTrainer

logging.basicConfig(
    format="[%(levelname)s|%(name)s] %(asctime)s >> %(message)s",
    # datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

task_manual_templates = {
    'sst2': '{"placeholder": "text_a"} It was {"mask"} .',
}

task_mix_templates = {
    'sst2': '{"placeholder": "text_a"} {"soft": "It was"} {"mask"} .',
    'cb': '{"placeholder": "text_a"} {"soft": "Question:"} {"placeholder": "text_b"}? Is it correct? {"mask"} .',
}

task_soft_templates = {
    'sst2': '{"placeholder": "text_a"} It was {"mask"} .',
}

task_ptuning_templates = {
    'sst2': '{"placeholder": "text_a"} {"soft"} It was {"mask"} .',
}

# https://github.com/THUDM/P-tuning/blob/main/PT-Fewshot/data_utils/task_pvps.py

task_verbalizers = {
    'sst2': {
        "negative": "terrible",
        "positive": "great",
        # "negative": ["bad", "terrible"],
        # "positive": ["good", "wonderful", "great"],
    }
}

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "result.json")
    json_string = json.dumps(results, indent=2, sort_keys=True) + "\n"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_string)

@dataclasses.dataclass
class PromptEvalConfig:
    method: str = 'prompt_tuning'
    model: str = 'bert'
    model_name_or_path: str = 'bert-base-uncased'
    seed: int = 42
    task: str = 'sst2'
    do_train: bool = True
    zero_shot: bool = True
    tune_plm: bool = False
    shot: int = 16 # -1 means full data
    metric_for_best_model: str = 'acc'

    loss_func: Optional[Callable] = None
    output_dir: str = 'result'
    fp16: bool = False # TODO

    # train config
    optimizer: str = 'adamw'
    warmup_steps: int = 0
    warmup_ratio: float = 0
    log_steps: int = 20
    eval_steps: int = 20
    max_steps: Optional[int] = None
    num_train_epochs: int = 10
    lr: float = 3e-5
    prompt_lr: Optional[float] = None
    weight_decay: float = 0.01
    train_batch_size: int = 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    early_stopping_patience: int = 8

    def __post_init__(self):
        if self.task == 'sst-2':
            self.task = 'sst2'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = f'{self.output_dir}/{self.method}/{self.task}'

    def save_to_json(self, json_path: Optional[str] = None):
        """Save the content of this instance in JSON format inside :obj:`json_path`."""
        os.makedirs(self.output_dir, exist_ok=True)

        json_path = json_path if json_path else os.path.join(self.output_dir, "prompteval_config.json")
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

class PromptEval:
    def __init__(self,
        config: PromptEvalConfig,
        template: Optional[Union[Template, str]] = None,
        verbalizer: Optional[Union[Verbalizer, dict]] = None
    ):
        self.config = config
        self.config.save_to_json()

        plm, tokenizer, model_config, WrapperClass = load_plm(self.config.model, self.config.model_name_or_path)
        self.plm = plm
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.WrapperClass = WrapperClass # TODO

        set_seed(self.config.seed)

        if self.config.task in ['sst2', 'cb']: # test split can not be evaluated
            splits = ['train', 'validation']
        else:
            splits = ['train', 'validation', 'test']

        if not self.config.do_train:
            splits = splits[-1:]

        if template is not None:
            if isinstance(template, Template):
                self.template = template
            elif isinstance(template, str):
                self.template = self.get_template(template_text=template)
            else:
                raise TypeError(f"`template` must be openprompt.Template or str,"
                                f"got {type(template)}.")
        else:
            self.template = self.get_template()

        if verbalizer is not None:
            if isinstance(verbalizer, Verbalizer):
                self.verbalizer = verbalizer
            elif isinstance(verbalizer, dict):
                self.verbalizer = self.get_verbalizer(verbalizer=verbalizer)
            else:
                raise TypeError(f"`verbalizer` must be openprompt.Verbalizer or dict,"
                                f"got {type(verbalizer)}.")
        else:
            self.verbalizer = self.get_verbalizer()
        
        self.prompt_model = PromptForClassification(
            plm=plm,
            template=self.template,
            verbalizer=self.verbalizer,
            freeze_plm=(not self.config.tune_plm))
    
        self.loss_func = self.config.loss_func if self.config.loss_func else torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.dataset, self.processor = load_data(self.config.task, splits=splits)
        self.dataset = load_data(self.config.task, splits=splits)

        if self.config.do_train and self.config.shot != -1:
            # true_few_shot_data
            sampler = FewShotSampler(num_examples_per_label=self.config.shot, also_sample_dev=True)
            self.dataset['train'], self.dataset['dev'] = sampler(self.dataset['train'], seed=self.config.seed)

        self.create_dataloader(self.dataset)

        self.trainer = PromptTrainer(
            model=self.prompt_model,
            train_dataloader=self.train_dataloader if hasattr(self, 'train_dataloader') else None,
            dev_dataloader=self.dev_dataloader if hasattr(self, 'dev_dataloader') else None,
            config=self.config,
            device=self.device
        )

    def get_template(self, template_text: Optional[str]=None) -> Template:
        method, task = self.config.method, self.config.task

        if method == 'manual':
            template = ManualTemplate(
                tokenizer=self.tokenizer,
                text=template_text if template_text is not None else task_manual_templates[task])
        elif method == 'warp':
            template = MixedTemplate(
                model=self.plm,
                tokenizer=self.tokenizer,
                text=template_text if template_text is not None else task_mix_templates[task])
        elif method == 'prompt_tuning':
            template = SoftTemplate(
                model=self.plm,
                tokenizer=self.tokenizer,
                num_tokens=20,
                initialize_from_vocab=True,
                text=template_text if template_text is not None else task_soft_templates[task])
        elif method == 'p_tuning':
            template = PtuningTemplate(
                model=self.plm,
                tokenizer=self.tokenizer,
                prompt_encoder_type="lstm",
                text=template_text if template_text is not None else task_ptuning_templates[task])
        else:
            raise NotImplementedError

        return template
    
    def get_verbalizer(self, verbalizer: Optional[dict]=None) -> Verbalizer:
        method, task = self.config.method, self.config.task

        task_classes = {
            'sst2': ['negative', 'positive']
        }

        if method in ['manual', 'prompt_tuning', 'p_tuning']:
            verbalizer = ManualVerbalizer(
                tokenizer=self.tokenizer,
                classes=task_classes[self.config.task],
                label_words=verbalizer if verbalizer is not None else task_verbalizers[task])
        elif method == 'warp':
            verbalizer = SoftVerbalizer(
                tokenizer=self.tokenizer,
                model=self.plm,
                classes=task_classes[self.config.task],
                label_words=verbalizer if verbalizer is not None else task_verbalizers[task])
        else:
            raise NotImplementedError

        # if method not in Verbalizers:
        #     raise NotImplementedError

        # verbalizer = Verbalizers[method](
        #     tokenizer=self.tokenizer,
        #     classes=['negative', 'positive'],
        #     label_words=verbalizer if verbalizer is not None else task_verbalizers[task])

        return verbalizer

    def train(self):
        self.trainer.train()

    def eval(self, eval_dataloader=None):
        eval_dataloader = eval_dataloader if eval_dataloader else self.dev_dataloader
        results = self.trainer.evaluate(eval_dataloader)
        # logger.info(results)
        return results
    
    def run(self):
        results_to_save = {}
        if self.config.zero_shot:
            logger.info("Zero-Shot Results:")
            results = self.eval(self.test_dataloader)
            logger.info(results)
            results_to_save["Zero-Shot Results"] = results

        if self.config.do_train:
            self.train()
            
            logger.info("Test Results:")
            results = self.eval(self.test_dataloader)
            logger.info(results)
            results_to_save["Test Results"] = results

        save_results(results_to_save, self.config.output_dir)

        return results

    def create_dataloader(self, dataset) -> None:
        if self.config.do_train:
            self.train_dataloader = PromptDataLoader(dataset=dataset["train"], template=self.template,
                tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, 
                max_seq_length=self.config.max_seq_length, decoder_max_length=3, 
                batch_size=self.config.train_batch_size, shuffle=True, teacher_forcing=False, 
                predict_eos_token=False, truncate_method="tail")

            dev_datasets = dataset.get('dev', dataset["validation"])
            if len(dev_datasets) == 0:
                dev_datasets = dataset["validation"]
            self.dev_dataloader = PromptDataLoader(dataset=dev_datasets, 
                template=self.template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, 
                max_seq_length=self.config.max_seq_length, decoder_max_length=3, 
                batch_size=self.config.eval_batch_size, shuffle=False, teacher_forcing=False, 
                predict_eos_token=False, truncate_method="tail")

        self.test_dataloader = PromptDataLoader(dataset=dataset.get('test', dataset["validation"]), 
            template=self.template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.WrapperClass, 
            max_seq_length=self.config.max_seq_length, decoder_max_length=3, 
            batch_size=self.config.eval_batch_size, shuffle=False, teacher_forcing=False, 
            predict_eos_token=False, truncate_method="tail")

