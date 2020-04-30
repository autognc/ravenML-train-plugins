import click

## Actual Option Definitions ##
comet_opt = click.option(
    '-c', '--comet', is_flag=True,
    help='Enable comet on this training run.'
)

name_opt = click.option(
    '-n', '--name', type=str, 
    help='First and Last name of user.'
)

comments_opt = click.option(
    '--comments', type=str, 
    help='Comments about the training.'    
)

model_opt = click.option(
    '--model-name', '-m', type=str,
    help='Name of model to be used for training.'
)

overwrite_local_opt = click.option(
    '--overwrite-local', '-o', is_flag=True,
    help='Overwrite files that may be in path specified.'
)

optimizer_opt = click.option(
    '--optimizer', type=str,
    help='Optimizer for training.'
)

use_default_config_opt = click.option(
    '-d', '--use-default-config', is_flag=True,
    help='Use default configuration for training'
)

hyperparameters_opt = click.option(
    '--hyperparameters', type=str,
    help='List of specified configurations for training'
)

# Maintain a list of the options for creating the condensed decorator
opts = [
    comet_opt, 
    name_opt, 
    comments_opt, 
    model_opt, 
    overwrite_local_opt, 
    optimizer_opt, 
    use_default_config_opt, 
    hyperparameters_opt
]
    
## Importable Option Decorator ##
def option_decorator(func):
    chain = opts[-1](func)
    for i in range(len(opts) - 2, -1, -1):
        chain = opts[i](chain)
    return chain
