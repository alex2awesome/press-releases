 python src/trainer.py \
     --model_name_or_path roberta-base \
     --dataset_name ../data/training_data_background_excluded.jsonl \
     --do_train \
     --do_eval \
     --output_dir /dev/shm/roberta-base__sentence-model__background-excluded \
     --overwrite_output_dir \
     --report_to wandb \
     --per_device_train_batch_size 1 \
     --per_device_eval_batch_size 1 \
     --model_type sentence \
     --evaluation_strategy steps \
     --context_layer transformer \
     --pooling_method attention \
     --eval_steps 100 \
     --freeze_layers 0 1 2 3

