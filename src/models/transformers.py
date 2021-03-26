import torch
from transformers import AutoModel
from transformers import AutoConfig
import numpy as np
import torch
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
import math


class TransformerBase(torch.nn.Module):
    def __init__(self, args):
        super(TransformerBase, self).__init__()
        transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True,
                                                        output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model, config=transformer_config)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.linear = torch.nn.Sequential(torch.nn.Linear(transformer_config.hidden_size, args.max_seq_len),
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(args.max_seq_len, args.num_labels))

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        attentions = output['attentions']
        pooled_output = torch.mean(output['last_hidden_state'], 1)
        # pooled_output = output['pooler_output'] this gives worse results
        predict = self.linear(self.dropout(pooled_output))
        return predict, attentions


class Attention(torch.nn.Module):
    """Scaled dot product attention."""

    def __init__(self, hidden_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.projection_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, atten_post):
        posts_attention_values = self.projection_layer(atten_post)
        posts_attention_weights = F.softmax(posts_attention_values.permute(0, 2, 1), dim=-1)

        del posts_attention_values
        torch.cuda.empty_cache()

        self_atten_output_post = torch.matmul(posts_attention_weights, atten_post)
        self_atten_output_post = self_atten_output_post.sum(dim=1).squeeze(1)

        return self_atten_output_post, posts_attention_weights


class SentenceTransformer(torch.nn.Module):
    def __init__(self, args):
        super(SentenceTransformer, self).__init__()
        transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True,
                                                        output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model, config=transformer_config)
        self.transformer_linear = torch.nn.Sequential(torch.nn.Linear(transformer_config.hidden_size, args.max_seq_len),
                                                      torch.nn.Tanh())
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.attention = args.attention

        if self.attention:
            self.attention_layer = Attention(hidden_dim=transformer_config.hidden_size)
            # self.linear = torch.nn.Sequential(torch.nn.Linear(transformer_config.hidden_size, args.num_labels),
            #                                   torch.nn.LogSoftmax(dim=1))
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(transformer_config.hidden_size, args.num_labels))

        else:
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(transformer_config.hidden_size, args.num_labels))

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_masks):
        # list of input_ids and attention mask
        post_encodings = []
        post_attentions = []
        for idx, input_id in enumerate(input_ids):
            input_id = input_id.squeeze(dim=1)  # reduce dimension
            attention_mask = attention_masks[idx].squeeze(dim=1)
            output = self.transformer(input_ids=input_id, attention_mask=attention_mask)
            post_encoding = self.mean_pooling(output['last_hidden_state'], attention_mask)
            post_encodings.append(post_encoding)
            # post_encodings.append(self.transformer_linear(torch.mean(output['last_hidden_state'], 1)))
            post_attentions.append(output['attentions'])

        post_encodings = torch.stack(post_encodings)

        if self.attention:
            post_attn, attn = self.attention_layer(post_encodings)
            predict = self.linear(self.dropout(post_attn))
            all_attentions = (post_attentions, attn)
        else:
            post_encodings = torch.mean(post_encodings, dim=1)
            predict = self.linear(self.dropout(post_encodings))
            all_attentions = (post_attentions, None)

        return predict, all_attentions


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
                outputs, attentions = model(input_ids, attention_mask)
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
                outputs, attentions = model(input_ids, attention_mask)

                _, predictions = torch.max(outputs.data, 1)
                preds.append(predictions.cpu().detach().numpy())
                targs.append(targets.cpu().detach().numpy())

        preds = np.asarray(preds).flatten()
        targs = np.asarray(targs).flatten()

        return preds, targs


class SentenceTrainer(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args)

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
                outputs, attentions = model(input_ids, attention_mask)
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
                outputs, attentions = model(input_ids, attention_mask)

                _, predictions = torch.max(outputs.data, 1)
                preds.append(predictions.cpu().detach().numpy())
                targs.append(targets.cpu().detach().numpy())

        preds = np.asarray(preds).flatten()
        targs = np.asarray(targs).flatten()

        return preds, targs


TRANSFORMER_MODELS = {
    'bert': TransformerBase,
    'att_bert': SentenceTransformer,
}

TRAINERS = {
    'bert': Trainer,
    'att_bert': SentenceTrainer
}
if __name__ == '__main__':
    pass
