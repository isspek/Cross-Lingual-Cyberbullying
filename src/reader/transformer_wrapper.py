from transformers import AutoTokenizer
from pathlib import Path
import torch
from src.reader.pan_hatespeech import AUTHOR_SEP, AUTHOR_ID
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.utils import RANDOM_SEED
from pathlib import Path
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
from tqdm import tqdm


class ExistTaskDataset(torch.utils.data.Dataset):
    pass


class PanHateSpeechTaskDataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, max_seq_len, ground_truth=None, mode='joined'):
        self.files = files
        self.ground_truth = ground_truth
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @staticmethod
    def process_text(text):
        text = text.replace('#URL#', "[URL]")
        text = text.replace('#HASHTAG#', "[HASHTAG]")
        text = text.replace('#USER#:', "[USER]")
        text = text.replace('#USER#', "[USER]")
        text = text.replace('RT', "[RT]")
        return text

    def __getitem__(self, item):
        selected_files = [self.files[item]]
        tokenized_texts = []
        labels = []
        for profile_file in selected_files:
            tree = ET.parse(profile_file)
            root = tree.getroot()
            labels.append(self.ground_truth[profile_file.stem])

            if self.mode == 'joined':
                for child in root:
                    posts = []
                    for ch in child:
                        posts.append(ch.text)
                content = ' '.join(posts)
                content = PanHateSpeechTaskDataset.process_text(content)
                tokenized_texts.append(content)

            elif self.mode == 'joined_post_aware':
                for child in root:
                    posts = []
                    for ch in child:
                        posts.append(f'[POSTSTART] {ch.text} [POSTEND]')
                content = ' '.join(posts)
                content = PanHateSpeechTaskDataset.process_text(content)
                tokenized_texts.append(content)

            elif self.mode == 'hierarchical':
                posts = []
                for child in root:
                    for ch in child:
                        posts.append(PanHateSpeechTaskDataset.process_text(ch.text))
                tokenized_texts.append(posts)

        if np.asarray(tokenized_texts).shape[1] == 1:
            encoding = self.tokenizer.encode_plus(tokenized_texts[0], add_special_tokens=True,
                                                  # Add '[CLS]' and '[SEP]'
                                                  max_length=self.max_seq_len,
                                                  padding='max_length',  # Pad & truncate all sentences.
                                                  truncation=True,
                                                  return_token_type_ids=False,
                                                  return_attention_mask=True,  # Construct attn. masks.
                                                  return_tensors='pt'  # Return pytorch tensors.
                                                  )
            return dict(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                labels=torch.LongTensor(labels)
            )

        else:
            input_ids = []
            attention_masks = []
            for idx, tokenized_text in enumerate(tokenized_texts[0]):
                encoding = self.tokenizer.encode_plus(tokenized_text, add_special_tokens=True,
                                                      # Add '[CLS]' and '[SEP]'
                                                      max_length=self.max_seq_len,
                                                      padding='max_length',  # Pad & truncate all sentences.
                                                      truncation=True,
                                                      return_token_type_ids=False,
                                                      return_attention_mask=True,  # Construct attn. masks.
                                                      return_tensors='pt'  # Return pytorch tensors.
                                                      )
                input_ids.append(encoding['input_ids'])
                attention_masks.append(encoding['attention_mask'])

            return dict(
                input_ids=torch.stack(input_ids),
                attention_mask=torch.stack(attention_masks),
                labels=torch.LongTensor(labels)
            )

    def __len__(self):
        return len(self.files)


class PANHateSpeechTaskDatasetWrapper:

    def create_cv_folds(self):
        kf = StratifiedKFold(n_splits=self.cv, random_state=RANDOM_SEED, shuffle=True)
        train_folds = []
        test_folds = []
        for train_index, test_index in kf.split(self.profile_files, list(self.ground_truth.values())):
            train_folds.append(train_index)
            test_folds.append(test_index)

        return train_folds, test_folds

    SPECIAL_TOKENS = {
        'joined': {'additional_special_tokens': ["[RT]", "[USER]", "[URL]", "[HASHTAG]"]},
        'hierarchical': {'additional_special_tokens': ["[RT]", "[USER]", "[URL]", "[HASHTAG]"]},
        'joined_post_aware': {
            'additional_special_tokens': ["[RT]", "[USER]", "[URL]", "[HASHTAG]", "[POSTSTART]", "[POSTEND]"]}
    }

    def __init__(self, args):
        self.cv = args.cv
        data_path = Path(args.data)
        labels_path = data_path / 'truth.txt'
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.special_tokens_dict = PANHateSpeechTaskDatasetWrapper.SPECIAL_TOKENS[args.input_mode]
        self.tokenizer.add_special_tokens(self.special_tokens_dict)
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
                self.dataset.append(
                    (PanHateSpeechTaskDataset(train_files, max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
                                              ground_truth=self.ground_truth, mode=args.input_mode),
                     PanHateSpeechTaskDataset(test_files, max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
                                              ground_truth=self.ground_truth,
                                              mode=args.input_mode)))



        else:
            # TODO for test files, the files without labels
            test_files = self.profile_files
            self.dataset = PanHateSpeechTaskDataset(test_files, max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
                                                    ground_truth=self.ground_truth,
                                                    mode=args.input_mode)


DATA_LOADERS = {

    'pan_hatespeech': PANHateSpeechTaskDatasetWrapper,
    'exist': ExistTaskDataset

}
