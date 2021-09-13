import click
from ravenml.train.options import kfold_opt, pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput

@click.group(help='Top level command group description')
@click.pass_context
@kfold_opt
def aclgan(ctx, kfold):
    pass
    
@aclgan.command()
@pass_train
@click.pass_context
def train(ctx, train: TrainInput):
    # If the context (ctx) has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train"
    # after training, create an instance of TrainOutput and return it
    result = TrainOutput()
    return result               # return result back up the command chain
