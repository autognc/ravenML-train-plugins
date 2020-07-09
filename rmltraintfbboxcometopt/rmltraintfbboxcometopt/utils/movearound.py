def prepare_for_training(
    bbox_cache: RMLCache,
    base_dir: Path, 
    data_path: Path, 
    arch_path: Path, 
    model_type: str, 
    metadata: dict,
    config: dict):
    """ Prepares the system for training.
    Creates artifact directory structure. Prompts user for choice of optimizer and
    hyperparameters. Injects hyperparameters into config files. Adds hyperparameters
    to given metadata dictionary.
    Args:
        bbox_cache (RMLCache): cache object for the bbox plugin
        base_dir (Path): root of training directory
        data_path (Path): path to dataset
        arch_path (Path): path to model architecture directory
        model_type (str): name of model type (i.e, ssd_inception_v1)
        metadata (dict): metadata dictionary to add fields to
        config (dict): plugin config from user provided config yaml
    Returns:
        bool: True if successful, False otherwise
    """
    # hyperparameter metadata dictionary
    hp_metadata = {}
    
    # create a data folder within our base_directory
    os.makedirs(base_dir / 'data')

    # copy object-detection.pbtxt from dataset and move into training data folder
    pbtxt_file = data_path / 'label_map.pbtxt'
    shutil.copy(pbtxt_file, base_dir / 'data')

    # calculate number of classes from pbtxt file
    with open(pbtxt_file, "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)
    
    # get num eval examples from file
    num_eval_file = data_path / 'splits/complete/train/test.record.numexamples'
    try:
        with open(num_eval_file, "r") as f:
            lines = f.readlines()
            num_eval_examples = int(lines[0])

    except:
        num_eval_examples = 1

    # create models, model, eval, and train folders
    model_folder = base_dir / 'models' / 'model'
    # model_folder = models_folder / 'model'
    # os.makedirs(models_folder)
    eval_folder = model_folder / 'eval'
    train_folder = model_folder / 'train'
    os.makedirs(model_folder)
    os.makedirs(eval_folder)
    os.makedirs(train_folder)
    
    # load optimizer choices and prompt for selection
    defaults = {}
    defaults_path = Path(os.path.dirname(__file__)) / 'model_defaults' / f'{model_type}_defaults.yml'
    with open(defaults_path, 'r') as stream:
        try:
            defaults = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    optimizer_name = config['optimizer'] if config.get('optimizer') else user_selects('Choose optimizer', defaults.keys())
    hp_metadata['optimizer'] = optimizer_name
    
    ### PIPELINE CONFIG CREATION ###
    # grab default config for the chosen optimizer
    default_config = {}
    try:
        default_config = defaults[optimizer_name]
    except KeyError as e:
        hint = 'optimizer name, optimizer not supported for this model architecture.'
        raise_parameter_error(optimizer_name, hint)
    
    # create custom configuration if necessary
    user_config = default_config
    if not config.get('use_default_config'):
        if config.get('hyperparameters'):
            user_config = _process_user_hyperparameters(user_config, config['hyperparameters'])
        else:
            user_config = _configuration_prompt(user_config)
        
    #_print_config('Using configuration:', user_config)
        
    # add to hyperparameter metadata dict
    for field, value in user_config.items():
        hp_metadata[field] = value
        
    # load template pipeline file
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    pipeline_file_name = f'{model_type}_{optimizer_name.lower()}.config'
    pipeline_path = cur_dir / 'pipeline_templates' / pipeline_file_name
    with open(pipeline_path) as template:
        pipeline_contents = template.read()
    
    # insert training directory path into config file
    # TODO: figure out what the hell is going on here
    if base_dir.name.endswith('/') or base_dir.name.endswith(r"\\"):
        pipeline_contents = pipeline_contents.replace('<replace_path>', str(base_dir))
    else:
        if os.name == 'nt':
            pipeline_contents = pipeline_contents.replace('<replace_path>', str(base_dir) + r"\\")
        else:
            pipeline_contents = pipeline_contents.replace('<replace_path>', str(base_dir) + '/')

    # place TF record files into training directory
    num_train_records = 0
    num_test_records = 0
    records_path = data_path / 'splits/complete/train'
    for record_file in os.listdir(records_path):
        if record_file.startswith('train.record-'):
            num_train_records += 1
            file_path = records_path / record_file
            shutil.copy(file_path, base_dir / 'data')

        if record_file.startswith('test.record-'):
            num_test_records += 1
            file_path = records_path / record_file
            shutil.copy(file_path, base_dir / 'data')

    # convert int to left zero padded string of length 5
    user_config['num_train_records'] = str(num_train_records).zfill(5)
    user_config['num_test_records'] = str(num_test_records).zfill(5)
            
    # insert rest of config into config file
    for key, value in user_config.items():
        formatted = '<replace_' + key + '>'
        pipeline_contents = pipeline_contents.replace(formatted, str(value))

    # insert num clases into config file
    pipeline_contents = pipeline_contents.replace('<replace_num_classes>', str(num_classes))

    # insert num eval examples into config file
    pipeline_contents = pipeline_contents.replace('<replace_num_eval_examples>', str(num_eval_examples))

    # output final configuation file for training
    #with open(model_folder / 'pipeline.config', 'w') as file:
    #    file.write(pipeline_contents)

    #TODO: we want to return pipeline_configs
    
    # update metadata and return success
    metadata['hyperparameters'] = hp_metadata

    return pipeline_contents
