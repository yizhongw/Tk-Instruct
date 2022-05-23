export openai_key="Your OpenAI api key"

data_dir=data/splits/default
task_dir=data/tasks
output_dir=output/gpt3
max_num_instances_per_eval_task=20

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
    --output_dir ${output_dir}/default/${engine}
python src/compute_metrics.py ${output_dir}/default/${engine}/gpt3_predictions.json
done

echo "xlingual instruction + 2 positive examples"
for engine in "text-davinci-001" "davinci"
do
echo $engine
python src/run_gpt3.py \
    --data_dir data/splits/xlingual/ \
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
    --output_dir ${output_dir}/xlingual/${engine}
python src/compute_metrics.py ${output_dir}/xlingual/${engine}/gpt3_predictions.json
done
