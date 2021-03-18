from src.utils import data_args
from argparse import ArgumentParser
from src.models.transformers import TRANSFORMER_MODELS
from src.reader.transformer_wrapper import DATA_LOADERS
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', help='Enter the directory to save the trained models')
    parser.add_argument('--pretrained_model', help='Pretrained Model of Transformers')
    parser.add_argument('--model', help='Enter model', choices=['bert'])
    parser.add_argument('--task', help='Enter model', choices=['exist', 'pan_hatespeech'])
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--cv', type=int, help='It perform n fold cross validation training')
    parser = data_args(parser)
    args = parser.parse_args()

    dataset_loader = DATA_LOADERS[args.task](args)

    if args.cv:
        cv_folds = dataset_loader.dataset

        for train_dataset, test_dataset in cv_folds:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
            model = TRANSFORMER_MODELS[args.model](args)
            model.train()
