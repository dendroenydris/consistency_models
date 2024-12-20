#!/bin/bash
#SBATCH --job-name=diffscale
#SBATCH --output=runs/%j_%x.out
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu

# Set environment variables for PyTorch Distributed
export MASTER_ADDR=$(hostname)   # Address of the master node
export MASTER_PORT=29500         # Communication port
export WORLD_SIZE=$SLURM_NTASKS  # Total number of processes
export LOCAL_RANK=$SLURM_LOCALID # Local rank on the current node
export RANK=$SLURM_PROCID        # Global rank across all nodes

# CONFIGURATION
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#  On Cluster           In Container
# ―――――――――――――――――――――――――――――――――――
#  $LOCAL_DATA          /data
#  $LOCAL_JOB_DIR       /mnt/output
#  ./code               /opt/code
# ___________________________________
# meta-data
DATE=$(date)
# set output directories
OUTPUTFOLDER="$SLURM_JOB_NAME-$SLURM_JOB_ID"
OUTPUTPATH_JOB="/opt/output"
SUBMIT_DIR=$(pwd)
OUTPUTPATH_LOCAL="$SUBMIT_DIR/runs/$OUTPUTFOLDER"
# create temporary output directory
# source "/etc/slurm/local_job_dir.sh"
export LOCAL_JOB_DIR=/data/local/jobs/${SLURM_JOB_ID}
mkdir -p "${LOCAL_JOB_DIR}/job_results"
APPTAINER_BINDPATH=".:/opt/code,./:/opt/submit,${LOCAL_JOB_DIR}/job_results:/opt/output"

# create singularity container via singularity build --force --fakeroot base_lightning.sif base_lightning.def
# PYTHON COMMAND needs to be handed after the job-name (stored in $cmd), example call of this script:
#       `sbatch ./submission.sh job_name python my_training.py resnet-50 MNIST --pretrained=True`
# NOTE this script request your python-script to have a --path_out parameter
# cmd="${@:1} --pth_out $OUTPUTPATH_JOB"

cmd="torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ${@:1} --pth_out $OUTPUTPATH_JOB"

# information about environmental variables and other meta data
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " VARIABLE                 VALUE                                                 "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " SLURM_JOB_ID             $SLURM_JOB_ID"
echo " SLURM_JOB_NAME           $SLURM_JOB_NAME"
echo " SLURM_JOB_PARTITION      $SLURM_JOB_PARTITION"
echo " SLURM_JOB_GPUS           $SLURM_JOB_GPUS"
echo " SLURMD_NODENAME          $SLURMD_NODENAME"
echo " USER                     $USER"
echo " DATE                     $DATE"
echo " OUTPUTFOLDER             $OUTPUTFOLDER"
echo "________________________________________________________________________________"

SCRIPT_DATASET_VARIABLE="--data_dir"
LOCAL_DATASET_DIR="dataset"

IN32_ZIP_PATH="$LOCAL_DATASET_DIR/IN32.zip"
if [ -f "$IN32_ZIP_PATH" ]; then
    echo "Copying and unzipping dataset to $LOCAL_JOB_DIR/data"
    mkdir -p "$LOCAL_JOB_DIR/data"
    cp "$IN32_ZIP_PATH" "$LOCAL_JOB_DIR/data/"
    unzip -q "$LOCAL_JOB_DIR/data/IN32.zip" -d "$LOCAL_JOB_DIR/data/"
    echo "Successfully unzipped dataset"
    export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${LOCAL_JOB_DIR}/data:/data/"

    # Check if /data/IN32_CM/train exists
    if [ ! -d "$LOCAL_JOB_DIR/data/IN-32_CM/train" ]; then
        echo "Error: /data/IN-32_CM/train directory does not exist after unzipping."
        exit 1
    fi

    cmd="$cmd $SCRIPT_DATASET_VARIABLE=/data/IN-32_CM/train"
else
    echo "IN32.zip not found in $LOCAL_DATASET_DIR"
    exit 1
fi

export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},$DATAPOOL1"
# cd $SUBMIT_DIR
echo "Successfully moving dataset"

# running the python script
cmd2="pip install -e git+https://github.com/dendroenydris/consistency_models.git#egg=consistency_models"

echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo "Running Command: $cmd2"
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
apptainer exec --nv def/environment_image.sif $cmd2

echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo "Completed Command: $cmd2"
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"

# Log metadata
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Local Rank: $LOCAL_RANK"
echo "Rank: $RANK"

echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo "Running Python-Command: $cmd"
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"

apptainer exec --nv def/environment_image.sif $cmd
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo "Completed Python-Command: $cmd"
echo "________________________________________________________________________________"

# copying results from local
mkdir -p $OUTPUTPATH_LOCAL
cp -r ${LOCAL_JOB_DIR}/job_results/* $OUTPUTPATH_LOCAL
rm -r ${LOCAL_JOB_DIR}/job_results

# also copy output
cp "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out" "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"

# information about the outputs of the script
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " CONTENTS                 PATH                                                  "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " job_results              $OUTPUTPATH_LOCAL"
echo " .out file                ${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
echo "________________________________________________________________________________"
