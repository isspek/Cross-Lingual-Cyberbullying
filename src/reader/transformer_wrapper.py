from transformers import AutoTokenizer
from pathlib import Path
import torch
from src.reader.pan_hatespeech import AUTHOR_SEP, AUTHOR_ID
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.utils import RANDOM_SEED


class ExistTaskDataset(torch.utils.data.Dataset):
    pass


class PanHateSpeechTaskDataset(torch.utils.data.Dataset):
    def __init__(self, files, ground_truth=None):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class PANHateSpeechTaskDatasetWrapper:

    def create_cv_folds(self):
        kf = StratifiedKFold(n_splits=self.cv, random_state=RANDOM_SEED, shuffle=True)
        train_folds = []
        test_folds = []
        for train_index, test_index in kf.split(self.profile_files, list(self.ground_truth.values())):
            train_folds.append(train_index)
            test_folds.append(test_index)

        return train_folds, test_folds

    def __init__(self, args):
        self.cv = args.cv
        data_path = Path(args.data)
        labels_path = data_path / 'truth.txt'

        self.profile_files = np.asarray([path for path in data_path.glob('*.xml')])

        self.ground_truth = {}
        with open(labels_path, 'r') as r:
            labels = r.readlines()
            for label in labels:
                label = label.split(AUTHOR_SEP)
                self.ground_truth[label[0]] = int(label[1])

        if self.cv:
            train_folds, test_folds = self.create_cv_folds()
            self.dataset = []

            for idx, train_fold in enumerate(train_folds):
                train_files = self.profile_files[train_fold]
                test_files = self.profile_files[test_folds[idx]]
                self.dataset.append((PanHateSpeechTaskDataset(train_files, self.ground_truth),
                                     PanHateSpeechTaskDataset(test_files, self.ground_truth)))



        else:
            # TODO for test files, the files without labels
            self.dataset = PanHateSpeechTaskDataset()


DATA_LOADERS = {

    'pan_hatespeech': PANHateSpeechTaskDatasetWrapper,
    'exist': ExistTaskDataset

}
