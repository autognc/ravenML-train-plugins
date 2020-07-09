import shortuuid
def prepare_one_train(base_dir, arch_path, pipeline_contents, config_update):

    #create folder for new model each time
    unique_name = 'model' + str(shortuuid.uuid())
    model_folder = base_dir / 'models' / unique_name
    eval_folder = base_dir / 'eval'
    train_folder = base_dir / 'train'
    os.makedirs(model_folder)
    os.makedirs(eval_folder)
    os.makedirs(train_folder)

    #DO WE PUT EVAL FOLDER AND TRAIN FOLDER HERE TOO
    #WAIT TO SEE WHAT EVAL/TRAIN FOLDERS ACTUALLY DO

    # copy model checkpoints to our train folder
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_folder = arch_path
    checkpoint0_folder = cur_dir / 'checkpoint_0'
    file1 = checkpoint_folder / 'model.ckpt.data-00000-of-00001'
    file2 = checkpoint_folder / 'model.ckpt.index'
    file3 = checkpoint_folder / 'model.ckpt.meta'
    file4 = checkpoint0_folder / 'model.ckpt-0.data-00000-of-00001'
    file5 = checkpoint0_folder / 'model.ckpt-0.index'
    file6 = checkpoint0_folder / 'model.ckpt-0.meta'
    shutil.copy2(file1, train_folder)
    shutil.copy2(file2, train_folder)
    shutil.copy2(file3, train_folder)
    shutil.copy2(file4, train_folder)
    shutil.copy2(file5, train_folder)
    shutil.copy2(file6, train_folder)

    # load starting checkpoint template and insert training directory path
    checkpoint_file = checkpoint0_folder / 'checkpoint'
    with open(checkpoint_file) as cf:
        checkpoint_contents = cf.read()
    checkpoint_contents = checkpoint_contents.replace('<replace>', str(train_folder))
    with open(train_folder / 'checkpoint', 'w') as new_cf:
        new_cf.write(checkpoint_contents)

    #update config file with new hyperparameters
    for key, value in config_update.items():
        formatted = '<replace_' + key + '>'
        pipeline_contents = pipeline_contents.replace(formatted, str(value))

    # output final configuation file for training
    with open(model_folder / 'pipeline.config', 'w') as file:
        file.write(pipeline_contents)

    return model_folder
    