#python -m src.models.baseline \
#--data data/pan21-author-profiling-training-2021-03-14 \
#--lang en \
#--model_output_dir trained_models

python -m src.experiment \
--data data/pan21-author-profiling-training-2021-03-14 \
--lang en \
--model bert \
--model_output_dir trained_models \
--pretrained_model bert-base-cased \
--data data/pan21-author-profiling-training-2021-03-14/en \
--task pan_hatespeech \
--cv 10