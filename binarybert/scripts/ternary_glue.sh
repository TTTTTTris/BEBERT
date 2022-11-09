#!/bin/bash

# Step 1: First train a half-sized ternary BERT model from the dynabert model checkpoint
# Tips:
# 1. If trained with data augmentation, please add --aug_train
# 2. For activation quantziation, use uniform quant for A8, with ACT2FN=gelu;
#    use lsq quant for A4, use lsq, with ACT2FN=relu to ensure non-negativity of LSQ asymmetric quantization


export TASK_NAME=MNLI
export wbits=2
export abits=4
export JOB_ID=Ternary_W${wbits}A${abits}
export GLUE_DIR=./glue_data
export TEACHER_MODEL_DIR=models/dynabert/${TASK_NAME}
export STUDENT_MODEL_DIR=models/dynabert/${TASK_NAME}

if [ $abits == 4 ]
then
act_quan_method=lsq
ACT2FN=relu
else
act_quan_method=uniform
ACT2FN=gelu
fi

export CUDA_VISIBLE_DEVICES=0
python quant_task_distill_glue.py \
    --data_dir ${GLUE_DIR} \
    --job_id ${JOB_ID} \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --eval_step 1000 \
    --num_train_epochs 6 \
    --ACT2FN ${ACT2FN} \
    --output_dir output_tws/${JOB_ID}/${TASK_NAME} \
    --kd_type two_stage \
    --task_name $TASK_NAME \
    --teacher_model ${TEACHER_MODEL_DIR} \
    --student_model ${STUDENT_MODEL_DIR} \
    --weight_bits ${wbits} \
    --weight_quant_method twn \
    --input_bits ${abits} \
    --input_quant_method ${act_quan_method} \
    --clip_lr 1e-4 \
    --learnable_scaling 2>&1 | tee -a nohup_out/${TASK_NAME}_ternary.out
