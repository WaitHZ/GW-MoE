model_name_or_path=
TASK_NAME=

dataset_config_name="en"
do_train=True
do_eval=True
max_source_length=256
predict_with_generate=True
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
weight_decay=0.1
compute_memory=True

per_device_train_batch_size=4
per_device_eval_batch_size=16
num_train_epochs=10
learning_rate=1e-5

encoder_train_use_gw=True
encoder_eval_use_gw=False
encoder_max_num_slots=8

decoder_train_use_gw=True
decoder_eval_use_gw=False
decoder_max_num_slots=1

gw_entropy_threshhold=1.6


seed=42
output_dir=./checkpoints_moe/switch-base-8/${TASK_NAME}-ft-${encoder_max_num_slots}-${decoder_max_num_slots}-${gw_entropy_threshhold}/${learning_rate}/${num_train_epochs}/32/${seed}

echo "${output_dir}"
mkdir -p ${output_dir}

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/log.out
fi

torchrun --nproc_per_node 8  --master_addr localhost  --node_rank 0  --master_port 12345  --nnodes 1 s2s_glue.py \
      --model_name_or_path ${model_name_or_path} \
      --output_dir ${output_dir} \
      --task_name ${TASK_NAME} \
      --eval_dataset_name ${TASK_NAME} \
      --test_dataset_name ${TASK_NAME} \
      --dataset_config_name ${dataset_config_name} \
      --eval_dataset_config_name ${dataset_config_name} \
      --test_dataset_config_name ${dataset_config_name} \
      --predict_with_generate ${predict_with_generate} \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --num_train_epochs ${num_train_epochs} \
      --weight_decay ${weight_decay} --learning_rate ${learning_rate} \
      --overwrite_output_dir \
      --do_train ${do_train} \
      --do_eval ${do_eval} \
      --weight_decay ${weight_decay} \
      --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm True \
      --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
      --max_source_length ${max_source_length} --compute_memory ${compute_memory} \
      --seed ${seed} \
      --decoder_train_use_gw ${decoder_train_use_gw} \
      --decoder_eval_use_gw ${decoder_eval_use_gw} \
      --decoder_max_num_slots ${decoder_max_num_slots} \
      --encoder_train_use_gw ${encoder_train_use_gw} \
      --encoder_eval_use_gw ${encoder_eval_use_gw} \
      --encoder_max_num_slots ${encoder_max_num_slots} \
      --gw_entropy_threshhold ${gw_entropy_threshhold} \
      --logging_steps 50
