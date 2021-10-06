NOTE: The schema.json and scheme.py are prototypes, not used currently.
# rmltraintfbbox
This plugin is used to interface with the tensorflow object detection api.   
It's main use is to train bounding box models on datasets created using the tfrecord plugin.

To install:
```bash
conda activate ravenml
pip install -e .
```

To train a model:
```bash
conda activate ravenml
ravenml train -c <path to config yml file> tf-bbox train
```
