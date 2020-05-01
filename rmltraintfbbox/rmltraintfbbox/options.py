from ravenml.options import verbose_opt, author_opt, comments_opt
from ravenml.train.options import comet_opt, model_name_opt, \
    optimizer_opt, use_default_config_opt, hyperparameters_opt

# define options order
opts = [
    verbose_opt,
    comet_opt,
    author_opt,
    comments_opt,
    model_name_opt,
    optimizer_opt,
    use_default_config_opt,
    hyperparameters_opt
]

## Importable Option Decorator ##
def option_decorator(func):
    chain = opts[-1](func)
    # must loop backwards so order of arguments in click
    # command def is same as order in opts list
    for i in range(len(opts) - 2, -1, -1):
        chain = opts[i](chain)
    return chain
