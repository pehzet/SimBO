import json
class Experiment:
    def __init__(self, experiment_dict: dict) -> None:
        # Init experiment from dict
        for key, value in experiment_dict.items():
            setattr(self, key, value)
        self.default_path = f"data/{self.experiment_name}.json"
    
    def to_dict(self):
        # Export class to dictionary, excluding functions and nested objects
        def is_function_or_class(obj):
            return callable(obj) or isinstance(obj, type)

        def process_value(value):
            if is_function_or_class(value):
                return None  # Exclude functions and classes
            if isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        class_dict = self.__dict__.copy()
        for key in list(class_dict.keys()):
            if is_function_or_class(class_dict[key]):
                del class_dict[key]
            else:
                class_dict[key] = process_value(class_dict[key])
        return class_dict

    def close_and_save(self, path):
        # Close experiment and save to path
        self.close()
        self.save(path)

    def close(self):
        self.status = "closed"
    
    def save(self, path=None):
        if path is None:
            path = self.default_path
        # Save experiment to path
        with open(path, 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=4))