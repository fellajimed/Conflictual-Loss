import munch
from pathlib import Path
import ruamel.yaml as yaml

# logger object
import logging
logger = logging.getLogger('main_all')


class MyMunch(munch.Munch):
    def __init__(self, dictionary):
        super().__init__(**dictionary)

    def __getattr__(self, key):
        if isinstance(key, str) and key.startswith('_'):
            return getattr(self, key, None)

        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            try:
                return self[key]
            except KeyError:
                logger.warning(f"{key=} not found - None will be returned")
                return None


class DictToDotNotation:
    def __init__(self, dictionary):
        self.values = munch.munchify(dictionary, factory=MyMunch)

    def __getattr__(self, key):
        return getattr(self.values, key, None)


class ConfigYaml:
    '''
    read a yaml file and convert it to dot notation
    '''

    def __init__(self, path_yaml):
        # read yaml file
        self.path = Path(path_yaml).resolve()
        self.is_valid = True

        with open(self.path) as stream:
            try:
                self.dict_config = yaml.safe_load(stream)
                self.config = DictToDotNotation(self.dict_config)

            except yaml.YAMLError as exc:
                self.is_valid = False
                logger.warning(f"error while reading the yaml file: {exc}")

    def pretty_print(self):
        return yaml.dump(self.dict_config, Dumper=yaml.RoundTripDumper)

    def update(self, new_dictionary, **kwargs):
        self.dict_config.update(new_dictionary)
        self.config = DictToDotNotation(self.dict_config, **kwargs)

    def write_config(self, filename):
        '''
        write the config to a file
        '''
        if self.is_valid:
            with open(filename, 'w') as f:
                yaml.round_trip_dump(self.dict_config, f,
                                     default_flow_style=False)


def dot_notation_to_dict(dot_not_obj):
    if isinstance(dot_not_obj, DictToDotNotation):
        return munch.unmunchify(dot_not_obj.values)

    return munch.unmunchify(dot_not_obj)


def get_item_recursively(dictionary, ordered_fields):
    """
    dictionary: dict
    ordered_fields: list/tuple of str

    return the value: dictionary[ordered_fields[0]][ordered_fields[1]] ...
    """
    if isinstance(ordered_fields, str):
        return dictionary.get(ordered_fields, None)
    elif len(ordered_fields) == 1:
        return dictionary.get(ordered_fields[0], None)
    elif ordered_fields[0] in dictionary:
        dictionary = dictionary[ordered_fields[0]]
        if isinstance(dictionary, dict):
            ordered_fields = ordered_fields[1:]
            return get_item_recursively(dictionary, ordered_fields)
        else:
            logger.warning(f"{dictionary} is not a dict "
                           f"- failed for {ordered_fields[:2]}")
            return None
    else:
        return None
