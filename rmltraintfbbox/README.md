NOTE: The schema.json and scheme.py are prototypes, not used currently.
# rmltraintfbbox
This plugin is used to interface with the tensorflow object detection api.   
It's main use is to train bounding box models on datasets created using the tfrecord plugin.
## Installation
To install cd into the same directory as this ReadME and follow these commands:
```bash
conda activate ravenml
pip install -e .
```
## Sample Config - .yml file
```yaml
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
  
## Config File Options
|Parameter |Type| Description                                                      |
|----------|---|------------------------------------------------------------------|
|metadata  | string|Fill out author information and description of the training      |
|verbose   | bool(True/False) |Print out verbose description of training informatoin       |
|comet     | string | Name of comet experiment. If not specified, comet will not be used               |
|model | string | name of the model to be trained (see table below for options)                                 |
|optimizer | string | name of the optimizer to be used   (see table below for options)                             |
|use_default_config | bool(True/False) | Use default hyperparameters   (see ./rmltraintfbbox/utils/model_defaults/<model_name>.yml for defaults) |
|hyperparameters | key/value pairs | Hyperparameters to modify and their values   (see ./rmltraintfbbox/utils/model_defaults/<model_name>.yml for defaults and hyperparameter options for each model/optimizer pair) |
  
## Model Options
|Model|Optimizer|Hyperparameter Options|
|---|---|---|
|ssd_mobilenet_v2_coco|Adam| { train_steps: 45000
|||batch_size: 24|
|||dropout_keep_probability: 0.6|
|||first_schedule_steps: 22000|
|||second_schedule_steps: 35000|
||| initial_learning_rate: 0.001|
|||first_schedule_lr: 0.0001|
|||second_schedule_lr: 0.00001|
## To train a model:
```bash
conda activate ravenml
ravenml train -c <path to config yml file> tf-bbox train
```
