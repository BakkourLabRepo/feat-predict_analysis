import importlib

def import_config(config_fname):
    """
    Import the specified analysis configuration.
    
    Arguments
    ---------
    config_fname : str
        The name of the configuration file to import from the
        configs directory.
    
    Returns
    -------
    analysis_config : dict
        The analysis configuration dictionary.
    """
    config_fname = config_fname.replace('.py', '')
    config_module_name = f"{config_fname}"
    try:
        config = importlib.import_module(
            f'configs.{config_module_name}'
            ).analysis_config
        return config
    except ModuleNotFoundError:
        print(f"Error: {config_module_name}.py not found.")