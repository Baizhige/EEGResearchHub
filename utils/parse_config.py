import configparser

def parse_config(config_file):
    """
    Parse the configuration file and return the configuration as a dictionary.

    Args:
    - config_file (str): File path of the configuration file.

    Returns:
    - config_dict (dict): Parsed configuration as a dictionary.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    config_dict = {
        # Learning rate, float type
        'lr': float(config['DEFAULT']['lr']),

        # Batch size, int type
        'batch_size': int(config['DEFAULT']['batch_size']),

        # Log ID, str type
        'log_id': str(config['DEFAULT']['log_id']),

        # Dataset name, str type
        'dataset_name': str(config['DEFAULT']['dataset_name']),

        # Model cache path, str type
        'model_cache': str(config['DEFAULT']['model_cache']),

        # Random seed, int type
        'random_seed': int(config['DEFAULT']['random_seed']),

        # Number of channels, int type
        'num_channel': int(config['DEFAULT']['num_channel']),

        # Number of classes, int type
        'num_class': int(config['DEFAULT']['num_class']),

        # Window length, int type
        'len_window': int(config['DEFAULT']['len_window']),

        # Model storage path, str type
        'model_root': str(config['DEFAULT']['model_root']),

        # Data loader, int type
        'num_workers': int(config['DEFAULT']['num_workers']),

        # CUDA availability, bool type
        'is_cuda': bool(config['DEFAULT']['is_cuda']),

        # Debug mode, bool type
        'is_debug': bool(config['DEFAULT']['is_debug']),

        # Maximum number of epochs for training, int type
        'n_epoch': int(config['DEFAULT']['n_epoch']),

        # Dataset root, str type
        'dataset_root': str(config['DEFAULT']['dataset_root']),

        # Record root, str type
        'record_root': str(config['DEFAULT']['record_root']),

        # Record name, str type
        'record_name': str(config['DEFAULT']['record_name']),
    }

    return config_dict
