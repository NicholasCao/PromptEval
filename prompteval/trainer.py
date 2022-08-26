import collections
import inspect
from sklearn.metrics import f1_score, accuracy_score
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer, EarlyStoppingCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.optimization import Adafactor, AdamW, get_linear_schedule_with_warmup

import numpy as np

logger = logging.get_logger(__name__)

class PromptTrainer:
    def __init__(self, model, train_dataloader, dev_dataloader, config, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.config = config
        self.device = device

        self.model.to(device)
        self.loss_func = self.config.loss_func if self.config.loss_func else torch.nn.CrossEntropyLoss()

        self.is_training = False

        os.makedirs(self.config.output_dir, exist_ok=True)

    def evaluate(
        self,
        eval_dataloader: Optional[Dataset] = None
    ) -> Dict[str, float]:

        if eval_dataloader is None:
            eval_dataloader = self.dev_dataloader

        self.model.eval()
        all_preds = []
        all_labels = []

        for step, inputs in enumerate(eval_dataloader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
            labels = inputs['label']

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro' if self.config.task in ['agnews', 'trec'] else 'binary')
        metrics = {"eval_acc": acc, "eval_f1": f1} 

        if self.is_training:
            self.log(metrics)
        
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.config.output_dir
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        # Good practice: save your training arguments together with the trained model
        # torch.save(self.config, os.path.join(output_dir, 'train_config.json'))
    
    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.config.warmup_steps if self.config.warmup_steps > 0 else math.ceil(num_training_steps * self.config.warmup_ratio)
        )
        return warmup_steps

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        model = self.model
        if self.config.tune_plm:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in model.plm.named_parameters() if (not any(nd in n for nd in no_decay))],
                    'weight_decay': self.config.weight_decay,
                    'lr': self.config.lr
                },
                {
                    'params': [p for n, p in model.plm.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.config.lr
                }
            ]
            # for warp SoftVerbalizer
            # The parameters of these part should be optimized (or freezed) together with the plm.
            if hasattr(model.verbalizer, 'group_parameters_1'):
                optimizer_grouped_parameters.append({
                    'params': model.verbalizer.group_parameters_1,
                    'lr': self.config.prompt_lr if self.config.prompt_lr else self.config.lr
                })
        else:
            optimizer_grouped_parameters = []
        
        optimizer_grouped_parameters.append({
            'params': [p for name, p in model.template.named_parameters() if 'raw_embedding' not in name],
            'lr': self.config.prompt_lr if self.config.prompt_lr else self.config.lr
        }) # note that you have to remove the raw_embedding manually from the optimization

        # for warp SoftVerbalizer
        if hasattr(model.verbalizer, 'group_parameters_2'):
            optimizer_grouped_parameters.append(
                {
                    'params': model.verbalizer.group_parameters_2,
                    'lr': self.config.lr
                }
            )

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.lr)

        self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps) 

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = inputs.to(self.device)
        logits = model(inputs)
        labels = inputs['label']
        loss = self.loss_func(logits, labels)

        loss.backward()

        return loss.detach()

    def train(self):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.

        """

        self.is_training = True
        self.early_stopping_patience_counter = 0

        # Data loader and number of training steps
        train_dataloader = self.train_dataloader
        model = self.model

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.config.max_steps:
            max_steps = self.config.max_steps
            num_train_epochs = self.config.max_steps // num_update_steps_per_epoch + int(
                self.config.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.config.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.config.num_train_epochs)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()

        # Train!
        num_examples = len(train_dataloader) * self.config.train_batch_size

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Train batch size = {self.config.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        tr_loss = torch.tensor(0.0).to(self.config.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        model.zero_grad()

        from tqdm import tqdm
        pbar = tqdm(total=max_steps, desc="Training")
        should_training_stop = False
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            steps_in_epoch = len(train_dataloader)

            inputs = None
            for step, inputs in enumerate(epoch_iterator):
                tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.config.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.config.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                self.config.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self._maybe_log_save_evaluate(tr_loss, model)
                    pbar.update(1)

                if (
                    self.state.global_step >= self.state.max_steps
                    or (self.early_stopping_patience_counter >= self.config.early_stopping_patience
                        and self.config.early_stopping_patience > 0)
                ):
                    should_training_stop = True
                    break
            
            if should_training_stop:
                break

        pbar.close()

        if self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
            self.model.load_state_dict(state_dict)

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        self.log(metrics)

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        self.state.save_to_json(os.path.join(self.config.output_dir, "trainer_state.json"))

        self.is_training = False
        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)
    
    def _maybe_log_save_evaluate(self, tr_loss, model):
        # Log
        if self.state.global_step % self.config.log_steps == 0:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.mean().item()
            # reset tr_loss to zero
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)

        metrics = None
        # Evaluate
        if self.state.global_step % self.config.eval_steps == 0:
            metrics = self.evaluate()
            logger.info(metrics)
            self._save_checkpoint(model, metrics=metrics)

    def _save_checkpoint(self, model, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # Determine the new best metric / best model checkpoint
        metric_to_check = self.config.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        # TODO
        # operator = np.greater if self.config.greater_is_better else np.less
        operator = np.greater if True else np.less
        if (
            self.state.best_metric is None
            or self.state.best_model_checkpoint is None
            or operator(metric_value, self.state.best_metric)
        ):
            output_dir = self.config.output_dir
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir

            # Only save model when it is the best one
            self.save_model(output_dir)

            # Save optimizer and scheduler
            # TODO
            # if self.config.save_optimizer_and_scheduler:
            if False:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)

            self.early_stopping_patience_counter = 0

            # Save the Trainer state
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            self.early_stopping_patience_counter += 1

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
