export openai_key="Your OpenAI api key"

data_dir=data/splits/default
task_dir=data/tasks
output_dir=output/gpt3_arxiv
max_num_instances_per_eval_task=20

# echo "instruction only"
# for engine in "davinci" # "ada" "curie" "davinci" "text-ada-001" "text-curie-001" "text-davinci-001"
# do
# echo $engine
# python src/run_gpt3.py \
#     --data_dir $data_dir \
#     --task_dir $task_dir \
#     --max_num_instances_per_task 1 \
#     --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --engine ${engine} \
#     --output_dir ${output_dir}/instruct_only/${engine}
# python src/compute_metrics.py ${output_dir}/instruct_only/${engine}/gpt3_predictions.json
# done

echo "instruction + 2 positive examples"
for engine in "davinci" "text-ada-001"
do
echo $engine
python src/run_gpt3.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir}/instruct_pos_2/${engine}
python src/compute_metrics.py ${output_dir}/instruct_pos_2/${engine}/gpt3_predictions.json
done

echo "multilingual instruction + 2 positive examples"
for engine in "text-davinci-001" "davinci"
do
echo $engine
python src/run_gpt3.py \
    --data_dir data/splits/multilingual/ \
    --task_dir $task_dir \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir}/multilingual/${engine}
python src/compute_metrics.py ${output_dir}/multilingual/${engine}/gpt3_predictions.json
done

# echo "instruction + 2 positive example + 2 negtive example"
# for engine in "ada" "curie" "davinci" "text-ada-001" "text-curie-001" "text-davinci-001"
# do
# echo $engine
# python src/run_gpt3.py \
#     --data_dir $data_dir \
#     --task_dir $task_dir \
#     --max_num_instances_per_task 1 \
#     --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
#     --add_task_definition True \
#     --num_pos_examples 1 \
#     --num_neg_examples 1 \
#     --add_explanation True \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --engine ${engine} \
#     --output_dir ${output_dir}/instruct_pos_2_neg_2/${engine}
# python src/compute_metrics.py ${output_dir}/instruct_pos_2_neg_2/${engine}/gpt3_predictions.json
# done