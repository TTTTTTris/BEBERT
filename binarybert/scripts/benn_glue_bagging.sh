#!/bin/bash

# Quantization without KD, refering to Bagging
# Tips: If trained with data augmentation, please add --aug_train

export TASK_NAME=QNLI
export wbits=1
export abits=4
export JOB_ID=Ternary_W${wbits}A${abits}
export GLUE_DIR=./glue_data
export TEACHER_MODEL_DIR=models/dynabert/${TASK_NAME}
export STUDENT_MODEL_DIR=output_Bag/Ternary_W2A4/${TASK_NAME}/kd_stage2/ 

if [ $abits == 4 ]
then
act_quan_method=lsq
ACT2FN=relu
else
act_quan_method=uniform
ACT2FN=gelu
fi

export CUDA_VISIBLE_DEVICES=0
python bagging_glue_train.py \
    --data_dir ${GLUE_DIR} \
    --job_id ${JOB_ID} \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --eval_step 100 \
    --num_train_epochs 6 \
    --ACT2FN ${ACT2FN} \
    --output_dir output_Bag/${JOB_ID}/${TASK_NAME} \
    --kd_type no_kd \
    --task_name $TASK_NAME \
    --teacher_model ${TEACHER_MODEL_DIR} \
    --student_model ${STUDENT_MODEL_DIR} \
    --weight_bits ${wbits} \
    --weight_quant_method bwn \
    --input_bits ${abits} \
    --input_quant_method ${act_quan_method} \
    --clip_lr 1e-4 \
    --learnable_scaling \
    --is_binarybert \
    --split 2>&1 | tee -a nohup_out/bagging_${TASK_NAME}.out