#!/usr/bin/env bash

##
# Author(s):        Carson Schubert (carson.schubert14@gmail.com)
# Date Created:     02/25/2019
#
# Install/uninstall script for raven plugins.
# NOTE: this script MUST be run from the ravenML-plugins root.
##


set -o errexit      # exit immediately if a pipeline returns non-zero status
set -o pipefail     # Return value of a pipeline is the value of the last (rightmost) command to exit with a non-zero status, 
                    # or zero if all commands in the pipeline exit successfully.
set -o nounset      # Treat unset variables and parameters other than the special parameters 
                    # ‘@’ or ‘*’ as an error when performing parameter expansion.

# echo "Checking for active raven conda environment..."

# grab available conda envs
# ENVS=$(conda env list | awk '{print $1}' )
# # attempt to source environment
# if [[ $ENVS = *"raven"* ]]; then
#    source activate raven
#    echo "Successfully activated raven environment."
# else 
#    echo "Error: please install the raven conda environment on your system."
#    exit
# fi;

# default uninstall and gpu to "d" so they are ignored when passed to install.sh scripts
# NOTE: this requires that install.sh scripts parse the d argument but simply ignore it
uninstall_flag=d
gpu_flag=d
# flag for running in a conda env (will check dependencies after operations)
conda_flag=0
# current prefix for requirements files (normal/cpu by default)
requirements_prefix="requirements"

while getopts "ugc" opt; do
    case "$opt" in
        u)
            uninstall_flag=u
            ;;
        g)
            gpu_flag=g
            requirements_prefix="requirements-gpu"
            ;;
        c)
            conda_flag=1
     esac
done

# loop through plugin directories
for f in * ; do
    if [ -d ${f} ]; then
        echo "Going on plugin $f..."
        cd $f
        ./install.sh -$uninstall_flag -$gpu_flag
        if [ "$uninstall_flag" = "u" ]; then
            echo "Cleaning up plugin dependencies..."
            pip uninstall -r $requirements_prefix.txt -y
        fi
        cd - > /dev/null
    fi
done

# ensure conda env still meets environment.yml file if conda flag set
if [ $conda_flag -eq 1 ]; then
    # ensure raven core environment dependencies are still met
    echo "Checking raven core dependencies..."
    conda env update -f environment.yml
fi
