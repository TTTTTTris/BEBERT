#!/bin/bash

export TASK_NAME=SST-2
export wbits=1
export abits=4
export JOB_ID=Ternary_W${wbits}A${abits}
export GLUE_DIR=./glue_data
export TEACHER_MODEL_DIR=models/dynabert/${TASK_NAME}
export STUDENT_MODEL_DIR=output_A/Ternary_W1A4/${TASK_NAME}/kd_stage2/ 
# output_A, B, C + _aug

if [ $abits == 4 ]
then
act_quan_method=lsq
ACT2FN=relu
else
act_quan_method=uniform
ACT2FN=gelu
fi

export CUDA_VISIBLE_DEVICES=0
python benn_glue_train_A.py \
    --data_dir ${GLUE_DIR} \
    --job_id ${JOB_ID} \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --eval_step 100 \
    --num_train_epochs 6 \
    --ACT2FN ${ACT2FN} \
    --output_dir output_A/${JOB_ID}/${TASK_NAME} \
    --kd_type two_stage \
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
    --do_eval 2>&1 | tee -a nohup_eval_out/eval_${TASK_NAME}_A.out