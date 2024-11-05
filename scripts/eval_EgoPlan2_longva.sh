MODEL_NAME="longva"
MODEL_WEIGHT="/group/40007/luqiu/Ego4D_benchmark_eval/Ego4D_eval_model/eval_ego4D_mllm_weight_ql/LongVA-7B-DPO"
VIDEO="/group/40007/public_datasets/Ego4D/v1_288p"
ANNOTATION='EgoPlan-Bench2.json'

PROJECT_ROOT="/group/40007/luqiu/Ego4D_benchmark_release"

export CUDA_VISIBLE_DEVICES=0

cd ${PROJECT_ROOT}
nohup python3 -u eval.py \
--model ${MODEL_NAME} \
--weight_dir ${MODEL_WEIGHT} \
--video_dir ${VIDEO} \
--anno_path ${ANNOTATION}\
> eval_multiple_choice_${MODEL_NAME}.log 2>&1 &