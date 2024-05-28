# Environment
For this project, `python=3.9` was used. We recommend using a conda environment.

## Conda environment
```bash
conda create -n CL_env python=3.9
conda activate CL_env
python -m pip install -r requirements.txt
```

## Train models
To train a model, all you have to do is provide a config file (yaml) `CONFIGURATION.yaml`:
```bash
python -m src.main CONFIGURATION.yaml
```
The script will create a folder for the logs and will save all checkpoint in the same folder.

# Experiments

## Training the models
Different options for training the models are suggested in [training.md](./training.md).

## Evaluation on the test set

`LOG_FOLDERS` is definined in the file [create_config_files.py](./create_config_files.py) (variable `logs_folder`). 

### Compute logits
```bash
python -m src.evaluations.compute_logits --id --ood --jobs 5 --batch-size 5000 LOG_FOLDERS
```
For MC-Dropout evaluations:
```bash
python -m src.evaluations.compute_logits --id --ood --jobs 5 --mc-dropout --n 20 --batch-size 5000 LOG_FOLDERS
```

### Aggregate results in a csv file
This will create a csv file `agg_results.csv`
```bash
python -m src.evaluations.aggregate_results --batch-size 5000 --jobs 10 --csv-file agg_results LOG_FOLDERS
```

### Figures
For MLP models, the results are plotted in new folder `figures`:
```bash
python -m src.evaluations.plot_results -f agg_results.csv -p figures -j 16 --y-axis len_train_set --y-type int --x-axis model_section_hidden_layers --x-type int --parent
```
