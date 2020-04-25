#!/usr/bin/env bash

##
# Author(s):        Carson Schubert (carson.schubert14@gmail.com)
# Date Created:     03/07/2019
#
# Install/uninstall script for ravenml TensorFlow Semantic Segmentation plugin.
##


set -o errexit      # exit immediately if a pipeline returns non-zero status
set -o pipefail     # Return value of a pipeline is the value of the last (rightmost) command to exit with a non-zero status, 
                    # or zero if all commands in the pipeline exit successfully.
set -o nounset      # Treat unset variables and parameters other than the special parameters 
                    # ‘@’ or ‘*’ as an error when performing parameter expansion.

# parse flags
install=1
requirements_prefix="requirements"
while getopts "ugd" opt; do
    case "$opt" in
        u)
            install=0
            ;;
        g)
            requirements_prefix="requirements-gpu"
            export RML_TF_SEMANTIC_GPU=1
            echo "-- GPU mode!"
            ;;
     esac
done

if [ $install -eq 1 ]; then
    echo "Installing..."
    pip-compile --output-file $requirements_prefix.txt $requirements_prefix.in
    pip install -r $requirements_prefix.txt
    pip install https://github.com/autognc/models/archive/deeplab-0.0.1.tar.gz\#subdirectory\=research/deeplab
    pip install -e .
    unset RML_TF_SEMANTIC_GPU=1
else
    # NOTE: this does NOT clean up after the plugin (i.e, leaves plugin dependenices installed)
    # To clean up, use the install_all.sh script at the root of the plugins/ directory
    echo "Uninstalling..."
    pip uninstall ravenml-tf-semantic -y
fi
