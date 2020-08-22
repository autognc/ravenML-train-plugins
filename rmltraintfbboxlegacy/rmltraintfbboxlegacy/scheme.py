import os
import yaml
import json
import jsonschema
from jsonschema import validate
from pathlib import Path
from schema import Schema, Optional, Use, And, Or

def prompt_model():
    pass
    
def prompt_optimizer():
    pass

def prompt_hypers():
    pass
    
def build_subschemas():
    pass
    
def validateJson(jsonData, schema):
    try:
        validate(instance=jsonData, schema=schema)
    except jsonschema.exceptions.ValidationError as er:
        print(er)
        return False
    return True

# load model choices from YAML
models_path = os.path.dirname(__file__) / Path('utils') / Path('model_info.yml')
with open(models_path, 'r') as stream:
    try:
        models = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

optimizer_choices = {}
for model in models.keys():
    model_type = models[model]['type']
    defaults_path = Path(os.path.dirname(__file__)) / Path('utils') / 'model_defaults' / f'{model_type}_defaults.yml'
    with open(defaults_path, 'r') as stream:
        try:
            defaults = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    optimizer_choices[model] = defaults.keys()

# print(optimizer_choices)

example = {"verbose": True, "model_name": "ssd_resnet50_v1_fpn", "optimizer": "Momentum"}

schema = Schema({
    Optional('verbose', default=False): bool,
    Optional('comet', default=None): str,
    # Optional('model_name', default=prompt_model): str,
    Optional('optimizer', default=prompt_optimizer): str,
    Optional('hyperparameters', default=prompt_hypers) : dict
}).json_schema('test')

schema["properties"].update({
    'model_name': dict({
        "enum": list(models.keys())
    })
})

# schema["properties"]["model_name"].update({
#     "default": prompt_model
# })

optimizers_and_hyperparamaters_list = []
for model in optimizer_choices.keys():
    optimizer_dict = {}
    optimizer_dict["if"] = dict({
        "properties": dict({
            "model_name": dict({
                "const": model
            })
        })
    })
    optimizer_dict["then"] = dict({
        "properties": dict({
            "optimizer": dict({
                "enum": list(optimizer_choices[model])
            })
        })
    })
    optimizers_and_hyperparamaters_list.append(optimizer_dict)
schema["allOf"] = optimizers_and_hyperparamaters_list

js = schema

with open('schema.json', 'w') as f:
    json.dump(js, f, indent=4)

with open('schema.json', 'r') as f:
    schema = json.load(f)
    print(validateJson(example, schema))
