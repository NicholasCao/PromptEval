from datasets import load_dataset
from openprompt.data_utils import PROCESSORS, InputExample

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "agnews": ("text", None),
}

# add '.' at the end of text
def check_text(text):
    text = text.strip()
    if not text.endswith('.'):
        text += ' .'
    return text

def get_raw_dataset(task, split):
    if task == 'sst2':
        return load_dataset('glue', 'sst2', split=split)
    elif task == 'rte':
        return load_dataset('glue', 'rte', split=split)
    elif task == 'agnews':
        return load_dataset('ag_news', 'default', split=split)
    elif task == 'mrpc':
        return load_dataset('glue', 'mrpc', split=split)
    else:
        raise NotImplementedError()


def load_data(task, splits=['train', 'validation', 'test']):
    dataset = {}
    sentence1_key, sentence2_key = task_to_keys[task]

    for split in splits:
        dataset[split] = []

        raw_dataset = get_raw_dataset(task, split=split)
        for example in raw_dataset:
            input_example = InputExample(
                # guid=example['idx'],
                text_a=check_text(example[sentence1_key]),
                text_b=check_text(example[sentence2_key]) if sentence2_key else '',
                label=int(example['label']))
            dataset[split].append(input_example)

    return dataset
