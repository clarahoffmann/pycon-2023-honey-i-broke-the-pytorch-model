

# train linear models
python train_models.py --save_dir_logger metrics_csv --name_logger linear --output_file_name linear_data --data_type linear
python train_models.py --save_dir_logger metrics_csv --name_logger linear_no_relu --output_file_name linear_data_no_relu --break_activations True --data_type linear

# train nonlinear models
python train_models.py --save_dir_logger metrics_csv --name_logger circle --output_file_name circle_data --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_no_relu --output_file_name circle_data_no_relu --break_activations True --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_dataloader_broken --output_file_name circle_dataloader_broken --break_dataloader True --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_frozen_weights --output_file_name circle_data_frozen_weights --freeze_weights True --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_frozen_bias --output_file_name circle_data_frozen_bias --freeze_bias True --data_type nonlinear


# train larger models with weightwatcher and different breakages
# weightwatcher only works with 3+ layers
python train_models.py --save_dir_logger metrics_csv --name_logger circle_large --output_file_name circle_large_data --data_type nonlinear --weight_watcher True
python run_ww.py --ckpt_path metrics_csv/circle_large/version_0/checkpoints/epoch=49-step=650.ckpt  --output_file_path weightwatcher_metrics/circle.csv

python train_models.py --save_dir_logger metrics_csv --name_logger circle_large_no_relu --output_file_name circle_large_no_relu --break_activations True --data_type nonlinear --weight_watcher True
python run_ww.py --ckpt_path metrics_csv/circle_large_no_relu/version_0/checkpoints/epoch=49-step=650.ckpt  --output_file_path weightwatcher_metrics/circle_no_relu.csv

python train_models.py --save_dir_logger metrics_csv --name_logger circle_large_dataloader_broken --output_file_name circle_large_dataloader_broken --break_dataloader True --data_type nonlinear --weight_watcher True
python run_ww.py --ckpt_path metrics_csv/circle_large_dataloader_broken/version_0/checkpoints/epoch=49-step=350.ckpt  --output_file_path weightwatcher_metrics/circle_dataloader_broken.csv

python train_models.py --save_dir_logger metrics_csv --name_logger circle_large_frozen_weights --output_file_name circle_large_frozen_weights --freeze_weights True --data_type nonlinear --weight_watcher True
python run_ww.py --ckpt_path metrics_csv/circle_large_frozen_weights/version_0/checkpoints/epoch=49-step=650.ckpt  --output_file_path weightwatcher_metrics/circle_frozen_weights.csv

python train_models.py --save_dir_logger metrics_csv --name_logger circle_large_frozen_bias --output_file_name circle_large_frozen_bias --freeze_bias True --data_type nonlinear --weight_watcher True
python run_ww.py --ckpt_path metrics_csv/circle_large_frozen_bias/version_0/checkpoints/epoch=49-step=650.ckpt  --output_file_path weightwatcher_metrics/circle_frozen_bias.csv

# run cleanlab experiment
python run_cleanlab.py --save_dir_logger metrics_csv --name_logger cleanlab --output_file_name cleanlab_data