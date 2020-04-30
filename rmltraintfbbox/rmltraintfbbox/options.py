import click

opts = []

## Actual Option Definitions ##
opts.append(click.option(
    '-c', '--comet', type=str,
    help='Create comet experiment with specified name on this training run.'
))

opts.append(click.option(
    '-a', '--author', type=str, 
    help='First and Last name of user.'
))

opts.append(click.option(
    '--comments', type=str, 
    help='Comments about the training.'    
))

opts.append(click.option(
    '-m', '--model-name', type=str,
    help='Name of model to be used for training.'
))

opts.append(click.option(
    '-o', '--overwrite-local', is_flag=True,
    help='Overwrite files that may be in path specified.'
))

opts.append(click.option(
    '--optimizer', type=str,
    help='Optimizer for training.'
))

opts.append(click.option(
    '--use-default-config', is_flag=True,
    help='Use default configuration for training'
))

opts.append(click.option(
    '--hyperparameters', type=str,
    help='List of specified configurations for training'
))

## Importable Option Decorator ##
def option_decorator(func):
    chain = opts[-1](func)
    for i in range(len(opts) - 2, -1, -1):
        chain = opts[i](chain)
    return chain
