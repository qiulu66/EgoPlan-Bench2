ANNOTATION='EgoPlan-Bench2.json'

PROJECT_ROOT="/group/40007/luqiu/Ego4D_benchmark_release"

cd ${PROJECT_ROOT}
nohup python3 -u cot.py \
--anno_path ${ANNOTATION}\
> eval_multiple_choice_cot.log 2>&1 &