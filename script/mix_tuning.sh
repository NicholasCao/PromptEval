# --num_train_epochs 10 \
# --lr 3e-5 \

python cli.py \
    --method mix_tuning \
    --model bert \
    --model_name_or_path bert-base-uncased \
    --seed 42 \
    --task sst2 \
    --do_train True \
    --zero_shot True \
    --tune_plm False \
    --shot 16 \
    --metric_for_best_model acc \
    --output_dir result \
    --optimizer adamw \
    --warmup_ratio 0.1 \
    --log_steps 20 \
    --eval_steps 20 \
    --max_steps 500 \
    --prompt_lr 1e-3 \
    --train_batch_size 4 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 128 \
    --early_stopping_patience 10
