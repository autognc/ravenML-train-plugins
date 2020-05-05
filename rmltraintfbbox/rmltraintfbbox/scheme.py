import os
import yaml
import json
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

# load model choices from YAML
models_path = os.path.dirname(__file__) / Path('utils') / Path('model_info.yml')
with open(models_path, 'r') as stream:
    try:
        models = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

optimizer_choices = {}
for model_type in models.keys():
    defaults_path = Path(os.path.dirname(__file__)) / 'model_defaults' / f'{model_type}_defaults.yml'
    with open(defaults_path, 'r') as stream:
        try:
            defaults = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    optimizer_choices[model_type] = defaults.keys()

input_schema = Schema({
    Optional('verbose', default=False): bool,
    Optional('comet', default=None): str,
    Optional('model_name', default=prompt_model): And(str, *models.keys()),
    Optional('optimizer', default=prompt_optimizer): And(str, 
    Optional('hyperparameters', default=prompt_hypers): {

    }
})

js = input_schema.json_schema('test')


with open('schema.json', 'w') as f:
    json.dump(js, f, indent=4)
