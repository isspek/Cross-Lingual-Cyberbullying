source venv/bin/activate

echo "$1/en"
python -m src.experiment \
--pan \
--model att_bert \
--pretrained_model trained_models/sentence-transformers/quora-distilbert-multilingual \
--tokenizer trained_models/sentence-transformers/quora-distilbert-multilingual \
--input_mode hierarchical \
--task pan_hatespeech \
--num_labels 2 \
--max_seq_len 32 \
--attention \
--model_file trained_models/quora-distilbert-multilingual-tokens_en_es_pan_epochs_5_max_32_lr_3e_5_with_attention_cv_5 \
--dropout 0.1 \
--data "$1/en" \
--test_batch_size 2 \
--lang en \
--output_dir "$2"

echo "$1/es"
python -m src.experiment \
--pan \
--model att_bert \
--pretrained_model trained_models/sentence-transformers/quora-distilbert-multilingual \
--tokenizer trained_models/sentence-transformers/quora-distilbert-multilingual \
--input_mode hierarchical \
--task pan_hatespeech \
--num_labels 2 \
--max_seq_len 32 \
--attention \
--model_file trained_models/quora-distilbert-multilingual-tokens_en_es_pan_epochs_5_max_32_lr_3e_5_with_attention_cv_5 \
--dropout 0.1 \
--data "$1/es" \
--test_batch_size 2 \
--lang es \
--output_dir "$2"