from src.utils import data_args
from argparse import ArgumentParser
from src.models.transformers import TRANSFORMER_MODELS, TRAINERS
from src.reader.transformer_wrapper import DATA_LOADERS
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np
import random
from src.utils import RANDOM_SEED
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', help='Enter the directory to save the trained models')
    parser.add_argument('--pretrained_model', help='Pretrained Model of Transformers')
    parser.add_argument('--tokenizer', help='Transformer Tokenizer')
    parser.add_argument('--model', help='Enter model', choices=['bert', 'att_bert'])
    parser.add_argument('--task', help='Enter model', choices=['exist', 'pan_hatespeech'])
    parser.add_argument('--output_dir')
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--cv', type=int, help='It perform n fold cross validation training')
    parser.add_argument('--input_mode', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--attention_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--cuda', action='store_true', help='Indicate whether you use cpu or gpu')
    parser = data_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_loader = DATA_LOADERS[args.task](args)

    if args.cv:
        cv_folds = dataset_loader.dataset

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

        results = {}
        for idx, (train_dataset, test_dataset) in enumerate(cv_folds):
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
            model = TRANSFORMER_MODELS[args.model](args)
            model.transformer.resize_token_embeddings(len(dataset_loader.tokenizer))

            # save model
            saved_folder = output_dir / f'{idx + 1}.pt'

            if not saved_folder.exists():
                trainer = TRAINERS[args.model](args)
                trained_model = trainer.train(dataloader=train_dataloader, model=model)
                torch.save(trained_model.state_dict(), saved_folder)

                preds, targs = trainer.test(dataloader=test_dataloader, model=trained_model)
                print(f'CLASSIFICATION REPORT FOLD {idx + 1}')
                print(classification_report(y_true=targs, y_pred=preds, digits=4))
                print(f'CONFUSION MATRIX {idx + 1}')
                print(confusion_matrix(y_true=targs, y_pred=preds))
                if args.cuda:
                    torch.cuda.empty_cache()
                del model

            # this part was just for validating if save / load method works
            model = TRANSFORMER_MODELS[args.model](args)
            trainer = TRAINERS[args.model](args)
            model.transformer.resize_token_embeddings(len(dataset_loader.tokenizer))
            if args.cuda:
                model.to(torch.device('cuda'))
            model.load_state_dict(torch.load(saved_folder))

            preds, targs = trainer.test(dataloader=test_dataloader, model=model)
            print(f'CLASSIFICATION REPORT FOLD {idx + 1} FROM SAVED MODEL')
            print(classification_report(y_true=targs, y_pred=preds, digits=4))
            print(f'CONFUSION MATRIX {idx + 1} FROM SAVED MODEL')
            print(confusion_matrix(y_true=targs, y_pred=preds))
            print('CV RESULTS')

            if 'f1_macro' not in results:
                results['f1_macro'] = []
            results['f1_macro'].append(f1_score(y_true=targs, y_pred=preds, average='macro'))

            if 'f1_micro' not in results:
                results['f1_micro'] = []
            results['f1_micro'].append(f1_score(y_true=targs, y_pred=preds, average='micro'))

            if 'f1_weighted' not in results:
                results['f1_weighted'] = []
            results['f1_weighted'].append(f1_score(y_true=targs, y_pred=preds, average='weighted'))

            if 'accuracy' not in results:
                results['accuracy'] = []
            results['accuracy'].append(accuracy_score(y_true=targs, y_pred=preds))

            if 'precision' not in results:
                results['precision'] = []
            results['precision'].append(precision_score(y_true=targs, y_pred=preds))

            if 'recall' not in results:
                results['recall'] = []
            results['recall'].append(recall_score(y_true=targs, y_pred=preds))

        print(f'{args.cv} Fold Results')
        for key, value in results.items():
            print(key)
            print(f'Mean: {np.mean(np.asarray(value))*100}, std: {np.std(np.asarray(value))*100}')
