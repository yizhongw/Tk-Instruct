echo "Copy_demo for English track"
python src/run_baselines.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 1 --max_num_instances_per_eval_task 100 --method copy_demo
echo "Copy_input for English track"
python src/run_baselines.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 0 --max_num_instances_per_eval_task 100 --method copy_input
echo "Copy_demo for x-lingual track"
python src/run_baselines.py --task_dir data/tasks --data_dir data/splits/multilingual --max_num_instances_per_task 1 --max_num_instances_per_eval_task 100 --method copy_demo
echo "Copy_input for x-lingual track"
python src/run_baselines.py --task_dir data/tasks --data_dir data/splits/multilingual --max_num_instances_per_task 0 --max_num_instances_per_eval_task 100 --method copy_input