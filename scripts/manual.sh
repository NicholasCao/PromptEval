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
elif [ $task == "mrpc" ]
then 
    max_seq_length=200
else
    printf "Unkonwn task '${task}' \n" 
    exit
fi

python cli.py \
    --method manual \
    --model roberta \
    --model_name_or_path roberta-large \
    --seed 42 \
    --task $task \
    --do_train False \
    --zero_shot True \
    --tune_plm False \
    --shot 16 \
    --max_seq_length $max_seq_length

# sst2 64
# rte 200
# agnews 128
