# Raven Training Plugins
Default training plugins for ravenML, and examples for making your own plugins.

Note that these plugins assume that all ravenML dependencies are present in your
virtual environment or on your system. They do not install any dependencies that
ravenML also depends on.

## Structure
All plugins are independent and separate Python packages with the following structure:
```
ravenml_package_name/                   # name of the package, with underscores
    ravenml_package_name/               # ''
        __init__.py             
        core.py                 # core of plugin - contains command group 
                                    entire plugin flows from
    install.sh                  # install script for the plugin
    requirements.in             # user created requirements file for CPU install
    requirements-gui.in         # user created requirements file for GPU install
    requirements.txt            # pip-compile created requirements file for CPU install
    requirements-gpu.txt        # pip-compile created requirements file for GPU install
    setup.py                    # python package setuptools
```
Additional files, directories, and modules can be created as needed. Just be sure to include
an `__init__.py` in every directory you create, and think in modules.

## Requirements Scheme
In order to support both GPU and CPU installs, each plugin will depend on
requirements files rather than packages in `install_requires` in `setup.py`. 
We use [pip-compile](https://github.com/jazzband/pip-tools) to maintain all requirements with
uninstall functionality. Developers should create **two files**:
1. requirements.in
2. requirements-gpu.in

These files are manually maintained and include all python packages which the plugin
depends on (such as tensorflow, pytorch, etc). Crucially, they do **not** contain 
these packages' dependencies. `pip-compile` is then used on the `.in` files to create
two files:
1. requirements.txt
2. requirements-gpu.txt

Which are created by the command:
```
pip-compile --out-file <prefix>.txt <prefix>.in
```

These files record *all* dependencies for anything listed in `requirements.in` and `requirements-gpu.in`, respectively.
They can then be used to do a clean uninstall of a plugin's dependencies without leaving the dependencies of those dependencies
(whew) dangling on the system.

### install.sh
An `install.sh` script should be written to handle all install logic (including updating pip-compile created requirements files) and ensure 
external dependenices (such as NVIDIA Drivers or additional packages) are met on install. When uninstalling, this script
should **only** handle the uninstall logic for the package itself, not its dependenices. The reason for the difference in 
behavior between install and uninstall is simple: Because many
plugins will have similar dependencies, which themselves share dependencies, we cannot safely remove
all of an individual plugin's dependenices without uninstalling all plugins. What this means for plugins in this directory:
- Plugins can be installed individually using their respective `install.sh` script.
- Plugins can also be installed en masse using `install_all.sh` at the root of this directory.
- Plugins can **exclusively** be uninstalled en masse using `install_all.sh` at the root of this directory.
If you uninstall a plugin individually using its `install.sh` script, you will **only** be uninstalling the plugin itself;
none of its dependencies will be cleaned up. This should be avoided to prevent creating a bloated environment.

All `install.sh` scripts should support two flags:
- `-u`: uninstall. Passed to uninstall the plugin itself.
- `-g`: GPU install. Can be paired with `-u` for uninstalling a GPU install.

### install_all.sh
Installs all plugins in this directory using their `install.sh` scripts. Mostly a convenience item when installing.
When uninstalling, this script should be used **exclusively** in place of any individual plugin's `install.sh` script.
It supports the same two flags as any `install.sh`, plus one additional flag:
- `-u`: uninstall. Passed to uninstall all plugins, including all plugin dependencies.
- `-g`: GPU install. Can be paired with `-g` for uninstalling a GPU install.
- `-c`: conda mode. Passed to indicate that an uninstall is occuring inside a conda environment defined
  by the `environment.yml` in the root of this repository. This `environment.yml` will always mirror the
  `environment.yml` in [ravenML](https://github.com/autognc/ravenML). When passed, `install_all.sh` will
  verify the environment against `environment.yml` once `install.sh -u` is run for each plugin. This guarantees
  that if any plugins share dependencies with ravenML core these dependencies remain met at the completion of the plugin uninstall. 

## Making a Plugin

Follow these steps to create a plugin.

### 1. Create file structure.
Every ravenML training plugin will begin with the following file structure:
```
ravenml_<plugin_name>/                   # name of the package, with underscores
    ravenml_<plugin_name>/               # ''
        __init__.py             
        core.py                 # core of plugin - contains top level command 
                                    group for entire plugin 
    install.sh                  # install script for the plugin
    requirements.in             # user created requirements file for CPU install
    requirements-gpu.in         # user created requirements file for GPU install
    setup.py                    # python package setuptools
```

We will go through each of these files individually.

#### Inner `ravenml_<plugin_name>/` directory
Contains the source code for the plugin itself. Inside are two files:
1. `__init__.py`: empty file which marks this at a python module.
2. `core.py`: core of the plugin where the top level command group is located. Go from the skeleton below:
```python
import click
from ravenml.train.options import kfold_opt, pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput

@click.group(help='Top level command group description')
@click.pass_context
@kfold_opt
def <plugin_name>(ctx, kfold):
    pass
    
@<plugin_name>.command()
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
```
`TrainInput` and `TrainOutput` are described in detail in the [Interfaces](#standard-interfaces) section.

#### `setup.py`
Contains setuptools code for turning this plugin into a python package. Go from the skeleton below:
```python
from setuptools import setup

setup(
    name='ravenml_<plugin_name>',
    version='0.1',
    description='Training plugin for ravenML',
    packages=['ravenml_<plugin_name>'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [ravenml.plugins.train]
        <plugin_name>=ravenml_<plugin_name>.core:<plugin_name>
    '''
)
```

#### `requirements.in` and `requirements-gpu.in`
Contain all plugin Python dependencies. These should be manually created and updated.
It is expected that these will largely overlap - however, keeping them separate is the
cleanest way of doing things. Write these files exactly as you would a normal `requirements.txt`.
There is no skeleton for these files. See [requirements](https://pip.pypa.io/en/stable/user_guide/#requirements-files) 
information here.

#### `install.sh`
See [install.sh](#installsh) section for description. Go from the skeleton below:
```shell
#!/usr/bin/env bash

set -o errexit      # exit immediately if a pipeline returns non-zero status
set -o pipefail     # Return value of a pipeline is the value of the last (rightmost) command to exit with a non-zero status, 
                    # or zero if all commands in the pipeline exit successfully.
set -o nounset      # Treat unset variables and parameters other than the special parameters 
                    # ‘@’ or ‘*’ as an error when performing parameter expansion.

# parse flags
install=1
requirements_prefix="requirements"
# NOTE: the "d" argument must be parsed in the getops call even though it is ignored.
# It acts as a default for install_all.sh to pass.
while getopts "ugd" opt; do
    case "$opt" in
        u)
            install=0
            ;;
        g)
            requirements_prefix="requirements-gpu"
            echo "-- GPU mode!"
            ;;
     esac
done

if [ $install -eq 1 ]; then
    echo "Installing..."
    pip-compile --output-file $requirements_prefix.txt $requirements_prefix.in
    pip install -r $requirements_prefix.txt
else
    # NOTE: this does NOT clean up after the plugin (i.e, leaves plugin dependenices installed)
    # To clean up, use the install_all.sh script at the root of the plugins/ directory
    echo "Uninstalling..."
    pip uninstall <plugin_name> -y
fi

```

### 2. Write plugin specific code.
Create additional files, directories, and modules as needed. Just be sure to include
an `__init__.py` in every directory you create, and think in modules.  

Consider creating a separate directory for each sub command group under the main one
named after the plugin and defined in `core.py`. These are structured as:
```
ravenml_<plugin_name>/
    ravenml_<plugin_name>/
        __init__.py
        core.py
        <command_group_name>/
            __init__.py
            commands.py
```

Go from the skeleton below for `commands.py`:
```python
import click

### OPTIONS ###
# put all local command options here


### COMMANDS ###
@click.group()
@click.pass_context
def <command_group_name>(ctx):
    pass

@<command_group_name>.command()
def <command_name>():
    click.echo('Sub command group command here!')
```

Within this directory you can create an `interfaces.py` file for any interfaces you want to expose
for the command group and an `options.py` file for any command group options you want to expose.

To import this sub command group in `ravenml_<plugin_name>/ravenml_<plugin_name>/core.py` you would put the following lines:
```python
from ravenml_<plugin_name>.<command_group_name>.commands import <command_group_name>

<plugin_name>.add_command(<command_group_name>)         # this assumes that the command group defined in core.py is named <plugin_name>
```

### 3. Install plugin
**NOTE** You must be inside the same Python environment as your ravenML installation when performing this operation.

Install the plugin using your installation script via `./install.sh`, using the `-g` flag if appropriate. At this point, if you run
`pip list` you should see your plugin listed, with the underscores in its named replaced by dashes.

At this point, your plugin should automatically load inside of Raven. Run `raven train list` to see 
all installed plugins and verify that yours appears.

### 4. (Optional) Test uninstalling plugin
It is a good idea to test that your plugin can be properly uninstalled as well. Recall that to uninstall a plugin,
you **must** use the `install_all.sh` script with the `-u` flag (and `-g` if appropriate).

Note that plugins downloaded and installed outside of this repository cannot be uninstalled using the 
`install_all.sh` script, as they are not tracked. There is no easy solution for this. It is up to the user to either leave plugin 
dependencies installed in the environment, write additional scripts to ensure plugin depenency removal does 
not impact other plugins, or some other manual solution.

## Best Practices
The `ravenml.utils.plugins` module contains useful functions for performing common training plugin tasks
such as prompting for basic metadata. **You should explore this module and use it as much as possible in any plugin you create for consistency.**

Some important best practices are outlined below.

### Dynamic Imports
Oftentimes plugins will depend on large libraries such as Tensorflow or PyTorch. Due to the way that Click loads plugins,
if you import these with standard statements at the top of your files they will be imported on every `ravenml` command, including
ones that do not utilize your plugin. This makes the entire CLI slow. To avoid this, a dynamic import function is provided in `ravenml.utils.plugins`.

However, **this function cannot be imported.** You must copy and paste the code into any file you wish to use it in due to the way
it interacts with the Python import system. You can see the function below (it is also commented inside `ravenml.utils.plugins`):
```python
# function is derived from https://stackoverflow.com/a/46878490
def _dynamic_import(modulename, shortname = None, asfunction = False):
    """ Function to dynamically import python modules into the global scope.

    Args:
        modulename (str): name of the module to import (ex: os, ex: os.path)
        shortname (str, optional): desired shortname binding of the module (ex: import tensorflow as tf)
        asfunction (bool, optional): whether the shortname is a module function or not (ex: from time import time)
        
    Examples:
        Whole module import: i.e, replace "import tensorflow"
        >>> _dynamic_import('tensorflow')
        
        Named module import: i.e, replace "import tensorflow as tf"
        >>> _dynamic_import('tensorflow', 'tf')
        
        Submodule import: i.e, replace "from object_detction import model_lib"
        >>> _dynamic_import('object_detection.model_lib', 'model_lib')
        
        Function import: i.e, replace "from ravenml.utils.config import get_config"
        >>> _dynamic_import('ravenml.utils.config', 'get_config', asfunction=True)
        
    """
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = importlib.import_module(modulename)
    else:        
        globals()[shortname] = getattr(importlib.import_module(modulename), shortname)
```

## Standard Interfaces
Two classes define the **standard interface** between ravenML core and training plugins:
- `TrainInput`
- `TrainOutput`

Import them with the following code (also seen in the [example](#inner-raven_plugin_name-directory) for `core.py`):
```python
from raven.train.interfaces import TrainInput, TrainOutput
```

### TrainInput
Class whose objects contain all necessary information for a training plugin to actually train. 
<!--- Add link to Sphinx documentation here in the future. --->

### TrainOutput
Class whose objects contain all necessary information for Raven core to process and save or upload training artifacts.
<!--- Add link to Sphinx documentation here in the future. --->
