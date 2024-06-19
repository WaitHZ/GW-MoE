model_name_or_path=
dataset_name=

do_train=True
do_eval=True
per_device_train_batch_size=8
per_device_eval_batch_size=2
num_train_epochs=10
weight_decay=0.1
learning_rate=5e-5

dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
predict_with_generate=True

source_prefix=summarize:
metric_for_best_model=rouge2

seed=42

gw_entropy_threshhold=1.8
decoder_max_num_slots=8
decoder_train_use_gw=True
decoder_eval_use_gw=False
encoder_max_num_slots=32
encoder_train_use_gw=True
encoder_eval_use_gw=False

SAVE=./checkpoints_moe/${model_name_or_path##*/}/${dataset_name}-ft-$gw_entropy_threshhold-$encoder_max_num_slots-$decoder_max_num_slots/${learning_rate}/64/${num_train_epochs}/$seed

echo "${SAVE}"
mkdir -p ${SAVE}

if [ ! -f ${output_dir}/log.out ];then
echo "The file doesn't exist."
else
rm -d ${output_dir}/${log_out}
fi

torchrun --nproc_per_node 8  --master_addr localhost  --node_rank 0  --master_port 12348  --nnodes 1  run_summarization.py \
      --model_name_or_path ${model_name_or_path} \
      --dataset_name ${dataset_name} \
      --do_train ${do_train} --do_eval ${do_eval} --overwrite_output_dir \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${per_device_eval_batch_size} \
      --output_dir ${SAVE} \
      --num_train_epochs ${num_train_epochs} --learning_rate ${learning_rate} \
      --weight_decay ${weight_decay} --metric_for_best_model ${metric_for_best_model} \
      --val_max_target_length 60 \
      --num_beams 6 --max_length 60 --min_length 10 --no_repeat_ngram_size 3 \
      --source_prefix ${source_prefix} \
      --evaluation_strategy ${evaluation_strategy} --save_strategy ${save_strategy} \
      --predict_with_generate ${predict_with_generate} \
      --seed ${seed} \
      --warmup_ratio 0.1 \
      --gw_entropy_threshhold $gw_entropy_threshhold \
      --decoder_max_num_slots $decoder_max_num_slots \
      --decoder_train_use_gw $decoder_train_use_gw \
      --decoder_eval_use_gw $decoder_eval_use_gw \
      --encoder_max_num_slots $encoder_max_num_slots \
      --encoder_train_use_gw $encoder_train_use_gw \
      --encoder_eval_use_gw $encoder_eval_use_gw \
      --logging_steps 50

