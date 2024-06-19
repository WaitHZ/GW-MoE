model_name_or_path=
dataset_name=


context_column=context
question_column=question
answer_column=answers
do_train=True
do_eval=True

per_device_train_batch_size=8
per_device_eval_batch_size=16

learning_rate=3e-5
num_train_epochs=10
max_seq_length=384
doc_stride=128

# GW method
gw_entropy_threshhold=1.8

encoder_train_use_gw=True
encoder_eval_use_gw=False
encoder_max_num_slots=16

decoder_train_use_gw=True
decoder_eval_use_gw=False
decoder_max_num_slots=2

dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
predict_with_generate=True

seed=42


output_dir=./checkpoints_moe/switch-base-8/${dataset_name}-ft-${encoder_max_num_slots}-${decoder_max_num_slots}-${gw_entropy_threshhold}/${learning_rate}/64/${num_train_epochs}/${seed}

echo "${output_dir}"
mkdir -p ${output_dir}


torchrun --nproc_per_node 8  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes 1 run_seq2seq_qa.py \
        --model_name_or_path ${model_name_or_path} \
        --seed ${seed} \
        --output_dir ${output_dir} \
        --dataset_name ${dataset_name} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --num_train_epochs ${num_train_epochs} \
        --overwrite_output_dir \
        --do_train ${do_train} --do_eval ${do_eval} \
        --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
        --evaluation_strategy ${evaluation_strategy} \
        --save_strategy ${save_strategy} \
        --predict_with_generate ${predict_with_generate}  \
        --gw_entropy_threshhold ${gw_entropy_threshhold} \
        --decoder_train_use_gw ${decoder_train_use_gw} \
        --decoder_eval_use_gw ${decoder_eval_use_gw} \
        --decoder_max_num_slots ${decoder_max_num_slots} \
        --encoder_train_use_gw ${encoder_train_use_gw} \
        --encoder_eval_use_gw ${encoder_eval_use_gw} \
        --encoder_max_num_slots ${encoder_max_num_slots}
