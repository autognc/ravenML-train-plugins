NOTE: The schema.json and scheme.py are prototypes, not used currently.
# rmltraintfbbox
This plugin is used to interface with the tensorflow object detection api.   
It's main use is to train bounding box models on datasets created using the tfrecord plugin.

To install cd into the same directory as this ReadME and follow these commands:
```bash
conda activate ravenml
pip install -e .
```
##Sample Config - .yml file
```
# This sample config contains all fields supported by ravenML core and the bbox plugin.
# Plugin specific configuration is located in the plugin field.

dataset: click_test
overwrite_local: True
artifact_path: '~/Desktop/test'
# options are:
#   'stop' to stop the instance
#   'terminate' to terminate the instance
#   any other string: keep instance running
# if this field is not set, the default is 'stop'
ec2_policy: stop
metadata:
    created_by: Carson Schubert
    comments: no thanks
plugin:
    verbose: true
    comet: false
    model: ssd_mobilenet_v2_coco
    optimizer: RMSProp
    # NOTE: if use_default_config is true, hyperparameters are IGNORED
    use_default_config: true
    hyperparameters:
        train_steps: 1000
```


To train a model:
```bash
conda activate ravenml
ravenml train -c <path to config yml file> tf-bbox train
```
