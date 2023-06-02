"""
@author : Tien Nguyen
@date   : 2023-04-26
"""

import utils
import constants

class Configurer(object):
    def __init__(
            self, config_file: str
        ) -> None:
        self.setup()
        self.parse(config_file)

    def setup(
            self
        ) -> None:
        root_dir = utils.get_cwd()
        self.root_dir, _ = utils.split_file_path(root_dir)
        self.data_dir = utils.join_path((self.root_dir, constants.DATA))
        self.preprocess_dir = utils.join_path((self.data_dir,\
                                                        constants.PREPROCESS))
        self.src      = utils.join_path((self.root_dir, constants.SRC))
        self.configs  = utils.join_path((self.src, constants.YAML_DIR))

    def parse(
            self, config_file
        ) -> None:
        configs = utils.read_yaml(utils.join_path((self.configs,\
                                                                config_file)))
        for group, value in configs.items():
            if not isinstance(value, dict):
                exec(f"self.{group} = value")
                continue
            for task, behavior in value.items():
                exec(f"self.{task} = behavior")
