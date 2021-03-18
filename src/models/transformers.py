import torch
from transformers import AutoModel
from transformers import AutoConfig
import numpy as np
import torch
from transformers.optimization import get_linear_schedule_with_warmup


class TransformerBase(torch.nn.Module):
    def __init__(self, args):
        super(TransformerBase, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.linear = torch.nn.Sequential(torch.nn.Linear(self.transformer_config.hidden_size, args.max_seq_len),
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(args.max_seq_len, args.num_labels))

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output['last_hidden_state'], 1)
        predict = self.linear(self.dropout(pooled_output))
        return predict


class Trainer:
    def __init__(self, args):
        self.args = args
        self.use_gpu = args.cuda

    def train(self, dataloader, model):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        num_total_steps = len(dataloader) * self.args.epochs
        num_warmup_steps = num_total_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps
        )
        if self.use_gpu:
            model.to(torch.device('cuda'))
        for epoch in range(self.args.epochs):
            model.train()
            total_sample = 0
            correct = 0
            loss_total = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].squeeze(1).cuda() if self.use_gpu else batch['input_ids'].squeeze(
                    dim=1)
                attention_mask = batch['attention_mask'].cuda() if self.use_gpu else batch['attention_mask']
                targets = batch['labels'].squeeze(1).cuda() if self.use_gpu else batch['labels'].squeeze(1)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions.cpu().detach().numpy() == targets.cpu().detach().numpy()).sum()
                total_sample += input_ids.shape[0]
                loss_step = loss.item()
                loss_total += loss_step
                print(f'epoch {epoch}, step {batch_idx}, loss {loss_step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()

            loss_total = loss_total / total_sample
            acc_total = correct / total_sample
            print(f'epoch {epoch}, step {batch_idx}, loss {loss_total}, acc {acc_total}')

        return model

    def test(self, dataloader, model):
        model.eval()
        with torch.no_grad():
            preds = []
            targs = []
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].squeeze(1).cuda() if self.use_gpu else batch['input_ids'].squeeze(
                    dim=1)
                attention_mask = batch['attention_mask'].cuda() if self.use_gpu else batch['attention_mask']
                targets = batch['labels'].squeeze(1).cuda() if self.use_gpu else batch['labels'].squeeze(1)
                outputs = model(input_ids, attention_mask)

                _, predictions = torch.max(outputs.data, 1)
                preds.append(predictions.cpu().detach().numpy())
                targs.append(targets.cpu().detach().numpy())

        preds = np.asarray(preds).flatten()
        targs = np.asarray(targs).flatten()

        return preds, targs


TRANSFORMER_MODELS = {
    'bert': TransformerBase,
}

TRAINERS = {
    'bert': Trainer
}
if __name__ == '__main__':
    pass
