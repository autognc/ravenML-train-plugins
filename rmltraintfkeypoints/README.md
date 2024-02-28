# TF Keypoints

## Setup
To install move into the rmltraintfkeypoints and follow these commands:

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
cp sample_configs/example_config.yaml train.yml
```

4. Modify the config variables, the only necessary change is the dataset.

## To Train a Model

1. Activate RavenML: 
```bash
conda activate ravenml
```

2. Start training:
```bash
ravenml train -c <path to config yml file> tf-keypoints train
```

## To Convert a Model

1. Create a conversion environment:
```bash
conda create --name keras-convert python=3.10.0 tensorflow tf2onnx onnx
```

2. Activate the environment:
```bash
conda activate keras-convert
```

3. Run the conversion:
```bash
python convert.py artifacts/phase_0/model.h5 model.onnx
```
