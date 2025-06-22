import logging

from ruamel.yaml import YAML


class YParams:
    """ Yaml file parser """

    def __init__(
            self,
            yaml_filename: str,
            config_name: str,
            print_params: bool = False
    ) -> None:
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        with open(yaml_filename) as _file:

            for key, val in YAML().load(_file)[config_name].items():
                val = None if val == 'None' else val

                self.params[key] = val
                self.__setattr__(key, val)

        if print_params:
            self.log()

    def __getitem__(self, key):
        """
        Retrieves a parameter's value by key.

        :param key: The key of the parameter to retrieve.
        :return: The value of the parameter.
        """

        return self.params[key]

    def __setitem__(self, key, val):
        """
        Sets a parameter's value.

        :param key: The key of the parameter to set.
        :param val: The value to set for the parameter.
        """

        self.params[key] = val

    def get(self, key, default=None):
        """
        Retrieves the value of a parameter if it exists, otherwise returns a default value.

        :param key: The key of the parameter to retrieve.
        :param default: The default value to return if the parameter key does not exist.
        :return: The value of the parameter or the default value.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.params.get(key, default)

    def log(self) -> None:
        """
        Logs the current configuration to the logging framework. Outputs all key-value pairs in the configuration.
        """
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))

        for key, val in self.params.items():
            logging.info(str(key) + ' ' + str(val))

        logging.info("---------------------------------------------------")

    def update(self, new_params):
        """
        Updates the internal parameter dictionary with new or modified key-value pairs.

        :param new_params: A dictionary containing new or updated parameters.
        """
        self.params.update(new_params)
        for key, val in new_params.items():
            self.__setattr__(key, val)
