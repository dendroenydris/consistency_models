#!/bin/bash
#SBATCH --job-name=diffscale
#SBATCH --output=runs/%j_%x.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G


# CONFIGURATION
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#  On Cluster           In Container
# ―――――――――――――――――――――――――――――――――――
#  $LOCAL_DATA          /data
#  $LOCAL_JOB_DIR       /mnt/output
#  ./code               /opt/code
# ___________________________________
# meta-data
DATE=`date`
# set output directories
OUTPUTFOLDER="$SLURM_JOB_NAME-$SLURM_JOB_ID"
OUTPUTPATH_JOB="/opt/output"
SUBMIT_DIR=`pwd`
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
cmd="${@:1} --pth_out $OUTPUTPATH_JOB"

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


# the variable the script expects for the path to the dataset
SCRIPT_DATASET_VARIABLE="--data_dir"
# path to where the dataset is stored locally on this cluster-node
LOCAL_DATASET_DIR="dataset"

IN32_ZIP_PATH="$LOCAL_DATASET_DIR/IN32.zip"
if [ -f "$IN32_ZIP_PATH" ]; then
    echo "Copying and unzipping dataset to $LOCAL_JOB_DIR/data"
    mkdir -p "$LOCAL_JOB_DIR/data"
    cp "$IN32_ZIP_PATH" "$LOCAL_JOB_DIR/data/"
    unzip -q "$LOCAL_JOB_DIR/data/IN32.zip" -d "$LOCAL_JOB_DIR/data/"
    # Confirm successful unzip
    echo "Successfully unzipped dataset"
    # Bind the unzipped data to /data
    export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${LOCAL_JOB_DIR}/data:/data/"
    
    # Check if /data/IN32_CM/train exists
    if [ ! -d "$LOCAL_JOB_DIR/data/IN-32_CM/train" ]; then
        echo "Error: /data/IN-32_CM/train directory does not exist after unzipping."
        exit 1
    fi
    
    # Set dataset variable
    cmd="$cmd $SCRIPT_DATASET_VARIABLE=/data/IN-32_CM/train"
else
    echo "IN32.zip not found in $LOCAL_DATASET_DIR"
    exit 1
fi

# copying dataset (if provided)
# if [ -d $LOCAL_DATASET_DIR ];
# then
#     echo "copying dataset to $LOCAL_JOB_DIR/data"
#     mkdir $LOCAL_JOB_DIR/data
#     cp -r $LOCAL_DATASET_DIR $LOCAL_JOB_DIR/data/dataset
#     export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${LOCAL_JOB_DIR}/data:/data/"
#     # setting dataset variable post-hoc
#     cmd="$cmd $SCRIPT_DATASET_VARIABLE=/data/dataset"
# fi
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},$DATAPOOL1"
# cd $SUBMIT_DIR
echo "Successfully moving dataset"

echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo "Running Command: $cmd2"
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"

# running the python script
cmd2="pip install -e git+https://github.com/dendroenydris/consistency_models.git#egg=consistency_models"
apptainer exec --nv def/environment_image.sif $cmd2

echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo "Completed Command: $cmd2"
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"

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
