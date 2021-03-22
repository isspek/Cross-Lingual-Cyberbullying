#python -m src.models.baseline \
#--data data/pan21-author-profiling-training-2021-03-14 \
#--lang en \
#--model_output_dir trained_models

#python -m src.experiment \
#--data data/pan21-author-profiling-training-2021-03-14/en \
#--lang en \
#--model bert \
#--model_output_dir trained_models \
#--pretrained_model bert-base-cased \
#--task pan_hatespeech \
#--cv 10 \
#--train_batch_size 2 \
#--test_batch_size 1 \
#--input_mode joined \
#--tokenizer bert-base-cased \
#--num_labels 2 \
#--lr 3e-5 \
#--max_seq_len 500 \
#--cuda \
#--epochs 5 \
#--output_dir trained_models/bert_base_cased_epochs_3_max_len_500_pan

#python -m src.experiment \
#--data data/pan21-author-profiling-training-2021-03-14/es \
#--lang es \
#--model bert \
#--model_output_dir trained_models \
#--pretrained_model dccuchile/bert-base-spanish-wwm-cased \
#--task pan_hatespeech \
#--cv 10 \
#--train_batch_size 2 \
#--test_batch_size 1 \
#--input_mode joined \
#--tokenizer dccuchile/bert-base-spanish-wwm-cased \
#--num_labels 2 \
#--lr 3e-5 \
#--max_seq_len 500 \
#--cuda \
#--epochs 5 \
#--output_dir trained_models/bert_base_cased_epochs_3_max_len_500_pan_es