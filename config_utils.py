
import os
import json
import pickle

def load_config(json_filename="config.json"):
    try:
        with open(json_filename, 'r') as f:
            config_data = json.load(f)
        print(f"Configuration loaded from {json_filename}:")
        print(config_data)
        return config_data
    except FileNotFoundError:
        print(f"Error: Configuration file {json_filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse {json_filename}.")
        return None

def cache_data(func):
    def wrapper(*args, **kwargs):
        cache_filename = kwargs.get('cache_filename')
        if cache_filename:
            cache_filename = os.path.join("data", os.path.basename(cache_filename))
            kwargs['cache_filename'] = cache_filename

        force_refresh = kwargs.get('force_refresh', False)
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args_dict = dict(zip(arg_names, args))
        db_path = kwargs.get('db_path') or args_dict.get('db_path')

        if cache_filename and os.path.exists(cache_filename) and not force_refresh:
            cache_mtime = os.path.getmtime(cache_filename)
            db_mtime = os.path.getmtime(db_path) if db_path and os.path.exists(db_path) else 0
            if cache_mtime > db_mtime:
                print(f"Loading data from cache: {cache_filename}")
                with open(cache_filename, 'rb') as f:
                    return pickle.load(f)

        data = func(*args, **kwargs)
        if cache_filename:
            with open(cache_filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Data cached to {cache_filename}")
        return data
    return wrapper
