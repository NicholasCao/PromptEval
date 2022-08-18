# --num_train_epochs 10 \
# --lr 3e-5 \

# raw
# [INFO|prompteval] 2022-08-18 10:49:47,101 >> Test Results:
# [INFO|prompteval] 2022-08-18 10:49:50,493 >> {'eval_acc': 0.7729357798165137, 'eval_f1': 0.7843137254901961}

# bbpl
# ../bbpl/result/bbpl-base
# [INFO|trainer] 2022-08-18 10:51:31,685 >> Loading best model from result/mix_tuning/sst2 (score: 0.84375).
# [INFO|prompteval] 2022-08-18 10:51:31,982 >> Test Results:
# [INFO|prompteval] 2022-08-18 10:51:35,352 >> {'eval_acc': 0.7408256880733946, 'eval_f1': 0.7580299785867238}
#    --warmup_ratio 0.1 \

python cli.py \
    --method warp \
    --model roberta \
    --model_name_or_path roberta-large \
    --seed 42 \
    --task sst2 \
    --do_train True \
    --zero_shot True \
    --tune_plm False \
    --shot 16 \
    --metric_for_best_model acc \
    --output_dir result \
    --optimizer adamw \
    --log_steps 20 \
    --eval_steps 20 \
    --warmup_steps 0 \
    --max_steps 300 \
    --prompt_lr 1e-3 \
    --train_batch_size 2 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 64 \
    --early_stopping_patience 10

# python cli.py \
#     --method mix_tuning \
#     --model bert \
#     --model_name_or_path ../bbpl/result/bbpl-base \
#     --seed 42 \
#     --task sst2 \
#     --do_train True \
#     --zero_shot True \
#     --tune_plm False \
#     --shot 16 \
#     --metric_for_best_model acc \
#     --output_dir result \
#     --optimizer adamw \
#     --warmup_ratio 0 \
#     --log_steps 20 \
#     --eval_steps 20 \
#     --max_steps 500 \
#     --prompt_lr 1e-3 \
#     --train_batch_size 4 \
#     --eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --max_seq_length 128 \
#     --early_stopping_patience 10
