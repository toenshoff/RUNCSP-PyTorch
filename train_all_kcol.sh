for k in {4..10}
do
	python train_maxcol.py --num_workers 4 --num_col ${k} --epochs 100 --model_dir models/maxcol/${k}-Col --data_path 'data/K-Col-Graphs/'${k}'-Col/*.dimacs'
done
