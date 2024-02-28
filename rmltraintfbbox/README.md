NOTE: The schema.json and scheme.py are prototypes, not used currently.
# rmltraintfbbox
This plugin is used to interface with the tensorflow object detection api.   
It's main use is to train bounding box models on datasets created using the tfrecord plugin.
## Setup
To install cd into the same directory as this ReadME and follow these commands:
1. Activate RavenML: 
```bash
conda activate ravenml
```

2. Install the necessary dependencies:
```bash
pip install -e .
```

3. Make a copy of the sample config:
```bash
cp sample_configs/bbox_config_all_fields.yaml train.yml
```

4. Modify the config variables, the only necessary change is the dataset, but other config options are shown below.

## To Train a Model

1. Activate RavenML: 
```bash
conda activate ravenml
```

2. Start training:
```bash
ravenml train -c <path to config yml file> tf-bbox train
```

## To Convert a Model

1. Create a conversion environment:
```bash
conda create --name myenv python=3.10.0 tensorflow tf2onnx
```

3. Move into folder with the saved model:
```bash
cd ~/.ravenML/train_tf-bbox/bbox_model_archs/ssd_mobilenet_v2_320x320_coco17_tpu-8
```

4. Convert the model to onnx.
```bash
python -m tf2onnx.convert --saved-model saved_model/ --output model.onnx
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
||||
||||
||||
||||

## To train a model:
```bash
conda activate ravenml
ravenml train -c <path to config yml file> tf-bbox train
```
