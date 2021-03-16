from transformers import AutoTokenizer
from pathlib import Path
import torch
from src.reader.pan_hatespeech import AUTHOR_SEP, AUTHOR_ID


class ExistTaskDataset(torch.utils.data.Dataset):
    pass


class PanHateSpeechTaskDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class PANHateSpeechTaskDatasetWrapper:

    def create_cv_folds(self):
        print(self.profile_files)
        pass

    def __init__(self, args):
        data_path = Path(args.data)
        labels_path = data_path / 'truth.txt'

        self.profile_files = [path for path in data_path.glob('*.xml')]

        self.ground_truth = {}
        with open(labels_path, 'r') as r:
            labels = r.readlines()
            for label in labels:
                label = label.split(AUTHOR_SEP)
                self.ground_truth[label[0]] = int(label[1])

        if args.cv:
            self.dataset = self.create_cv_folds()
        else:

            self.dataset = PanHateSpeechTaskDataset()

        self.profile_files = data_path.glob('*.xml')


DATA_LOADERS = {

    'pan_hatespeech': PANHateSpeechTaskDatasetWrapper,
    'exist': ExistTaskDataset

}
