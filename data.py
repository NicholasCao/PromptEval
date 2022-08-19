from datasets import load_dataset
from openprompt.data_utils import PROCESSORS, InputExample

def load_data(task, splits=['train', 'validation', 'test']):
    if task == 'sst2':
        dataset = {}
        for split in splits:
            dataset[split] = []

            raw_dataset = load_dataset('glue', 'sst2', split=split)
            for example in raw_dataset:
                text_a = example['sentence'].strip()
                if not text_a.endswith('.'):
                    text_a += ' .'
                input_example = InputExample(guid=example['idx'], text_a=text_a, label=int(example['label']))
                dataset[split].append(input_example)
    # elif task == 'cb':
    #     # input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
    else:
        raise NotImplementedError()

    return dataset
