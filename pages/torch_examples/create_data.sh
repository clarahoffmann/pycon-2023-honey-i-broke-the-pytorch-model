

# train linear models
python train_models.py --save_dir_logger metrics_csv --name_logger linear --output_file_name linear_data --data_type linear
python train_models.py --save_dir_logger metrics_csv --name_logger linear_no_relu --output_file_name linear_data_no_relu --break_activations True --data_type linear

# train nonlinear models
python train_models.py --save_dir_logger metrics_csv --name_logger circle --output_file_name circle_data --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_no_relu --output_file_name circle_data_no_relu --break_activations True --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_dataloader_broken --output_file_name circle_dataloader_broken --break_dataloader True --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_frozen_weights --output_file_name circle_data_frozen_weights --freeze_weights True --data_type nonlinear
python train_models.py --save_dir_logger metrics_csv --name_logger circle_frozen_bias --output_file_name circle_data_frozen_bias --freeze_bias True --data_type nonlinear