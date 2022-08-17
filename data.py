from datasets import load_dataset
from openprompt.data_utils import PROCESSORS, InputExample

def load_data(task, splits=['train', 'validation', 'test']):
    if task == 'sst2':
        dataset = {}
        for split in splits:
            dataset[split] = []

            raw_dataset = load_dataset('glue', 'sst2', split=split)
            for data in raw_dataset:
                input_example = InputExample(guid=data['idx'], text_a=data['sentence'], label=int(data['label']))
                dataset[split].append(input_example)
    # elif task == 'cb':
    #     # input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
    else:
        raise NotImplementedError()

    return dataset
