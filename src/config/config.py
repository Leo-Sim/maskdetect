
import yaml
import os

class Config:
    def __init__(self, config_path=None):

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        self.config_path = config_path
        self.config = self._load_yaml()

    def _load_yaml(self):
        with open(self.config_path, "r") as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                print(f"Error loading YAML file: {e}")
                return {}

    # settings
    def get_training_config(self):
        return self.config.get("training", {})

    def get_model_config(self):
        return self.config.get("model", {})

    def get_data_config(self):
        return self.config.get("data", {})

    def get_batch_size(self):
        return self.get_training_config().get("batch_size", 32)

    def get_momentum(self):
        return self.get_training_config().get("momentum", 0.9)

    def get_image_size(self):
        size = self.get_training_config().get("image_size", (64, 64))
        return tuple(map(int, size.split(',')))

    def get_learning_rate(self):
        return self.get_training_config().get("learning_rate", 0.001)

    def get_epochs(self):
        return self.get_training_config().get("epoch_num", 10)

    def get_class_num(self):
        return self.get_data_config().get('class_num')

    def get_dataset_path1(self):
        return self._get_root_absolute_path(self.get_data_config().get('dataset_path1'))

    def _get_root_absolute_path(self, relative_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        return os.path.join(base_dir, relative_path)


config = Config()
print(config.get_batch_size())
print(config.get_training_config())
print('ddd   ',config.get_class_num())

