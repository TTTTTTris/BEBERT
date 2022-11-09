FT_BERT_BASE_DIR="./models/dynabert/qqp"
GENERAL_TINYBERT_DIR="./results/BiBERT/qqp" # evaluate
# GENERAL_TINYBERT_DIR="./models/dynabert/qqp" # train

TASK_DIR="./gule_data"
TASK_NAME="qqp"

OUTPUT_DIR="./results/BiBERT/qqp"

CUDA_VISIBLE_DEVICES=0  python eval_bibert.py \
            --data_dir $TASK_DIR \
            --teacher_model $FT_BERT_BASE_DIR \
            --student_model $GENERAL_TINYBERT_DIR \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --seed 42 \
            --learning_rate 1e-4 \
            --weight_bits 1 \
            --embedding_bits 1 \
            --input_bits 1 \
            --batch_size 32 \
            --pred_distill \
            --intermediate_distill \
            --value_distill \
            --key_distill \
            --query_distill \
            --do_eval \
            --save_fp_model 2>&1 | tee -a nohup_eval_out/${TASK_NAME}.out
