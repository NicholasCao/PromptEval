#    --warmup_ratio 0.1 \

python cli.py \
    --method warp \
    --model roberta \
    --model_name_or_path roberta-large \
    --seed 10 \
    --task rte \
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
    --max_steps 500 \
    --lr 3e-4 \
    --prompt_lr 5e-3 \
    --train_batch_size 2 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 64 \
    --early_stopping_patience 10
