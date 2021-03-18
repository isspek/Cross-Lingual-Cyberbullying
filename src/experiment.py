from src.utils import data_args
from argparse import ArgumentParser
from src.models.transformers import TRANSFORMER_MODELS, TRAINERS
from src.reader.transformer_wrapper import DATA_LOADERS
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import random
from src.utils import RANDOM_SEED

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', help='Enter the directory to save the trained models')
    parser.add_argument('--pretrained_model', help='Pretrained Model of Transformers')
    parser.add_argument('--tokenizer', help='Transformer Tokenizer')
    parser.add_argument('--model', help='Enter model', choices=['bert'])
    parser.add_argument('--task', help='Enter model', choices=['exist', 'pan_hatespeech'])
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--cv', type=int, help='It perform n fold cross validation training')
    parser.add_argument('--input_mode', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cuda', action='store_true', help='Indicate whether you use cpu or gpu')
    parser = data_args(parser)
    args = parser.parse_args()

    dataset_loader = DATA_LOADERS[args.task](args)

    if args.cv:
        cv_folds = dataset_loader.dataset

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        for idx, (train_dataset, test_dataset) in enumerate(cv_folds):
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
            model = TRANSFORMER_MODELS[args.model](args)
            model.transformer.resize_token_embeddings(len(dataset_loader.tokenizer))
            trainer = TRAINERS[args.model](args)
            trained_model = trainer.train(dataloader=train_dataloader, model=model)
            preds, targs = trainer.test(dataloader=test_dataloader, model=trained_model)
            print(f'CLASSIFICATION REPORT FOLD {idx + 1}')
            print(classification_report(y_true=targs, y_pred=preds, digits=4))
            print(f'CONFUSION MATRIX {idx + 1}')
            print(confusion_matrix(y_true=targs, y_pred=preds))
            if args.cuda:
                torch.cuda.empty_cache()
            del model
