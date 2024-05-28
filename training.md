# Option 1: One machine - sequential training

Here is a suggestion to train the models:
```bash
# path of the root of the code
EXP_DIR="$(dirname "$(readlink -f "$0")")"
CONFIG_FILES=$(find "$EXP_DIR"/mlp -type f -name "exp*.yaml")

for file in $CONFIG_FILES; do python -m src.main $file; done
```

> **_NOTE:_** if multiple GPUs are available, the training for each file is done on all of them (dataparallel is used)

# Option 2: One machine - parallel training
When having access to multiple GPUs on **a single machine**, it is possible to use <a href="https://www.gnu.org/software/parallel/">GNU Parallel</a>:

GNU Parallel could be installed using `apt-get`:
```bash
sudo apt-get install parallel
```

Here is a suggestion to train the models:
```bash
# path of the root of the code
EXP_DIR="$(dirname "$(readlink -f "$0")")"
CONFIG_FILES=$(find "$EXP_DIR"/mlp -type f -name "exp*.yaml")

# number of GPUs available (default `0`)
N_GPUs=$(nvidia-smi --list-gpus | wc -l || 0)

echo $CONFIG_FILES | parallel --delay 2 -j $N_GPUs CUDA_VISIBLE_DEVICES='$(({%} - 1))' python -m src.main {}
```

> **_NOTE:_** it is possible to shuffle the order of the config files `echo $CONFIG_FILES | shuf | parallel ...`

* `--delay 2`: add a delay of 2 seconds between the executions since the log files are created based on the time
* For this example, each configuration is executed using only **one** GPU: `CUDA_VISIBLE_DEVICES='$(({%} - 1))'`

# Option 3: Using a cluster
Even if the gpus are not all in the same machine, it is still possible to run the experiments using GNU Parallel:
```bash
# path of the root of the code
EXP_DIR="$(dirname "$(readlink -f "$0")")"
CONFIG_FILES=$(find "$EXP_DIR"/mlp -type f -name "exp*.yaml")

conda activate CL_env
PYTHON_PATH=$(which python)

echo $CONFIG_FILES | parallel --slf workers-ssh --delay 2 $PYTHON_PATH -m src.main {}
```
* The file `workers-ssh` is the ssh login file. One can also start the file with the GPU devices to use one GPU per training.
* The variable `PYTHON_PATH` is important so that the code is not executed with the default python of the machine.
