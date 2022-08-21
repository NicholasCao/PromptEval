getopts "t:" opt
if 	[ $opt != "t" ]
then
    printf "[Usage] `basename $0` -t <task_name> \n" 
    exit
fi

task=${OPTARG}

# get default max_seq_length by task name
if [ $task == "sst2" ]
then
    max_seq_length=64
elif [ $task == "rte" ]
then 
    max_seq_length=200
elif [ $task == "agnews" ]
then 
    max_seq_length=128
else
    printf "Unkonwn task '${task}' \n" 
    exit
fi

for seed in 10 42 100
do
    python cli.py \
        --method warp \
        --model roberta \
        --model_name_or_path roberta-large \
        --seed $seed \
        --task $task \
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
        --max_seq_length $max_seq_length \
        --early_stopping_patience 10
done
