class ScrapperConfig:
    def __init__(self, file_with_classes=None, target_path='data/',
                 path_to_driver=None, download_data=False, path_to_data=None):
        self.file_with_classes = file_with_classes
        self.path_to_driver = path_to_driver
        self.target_path = target_path
        self.download_data = download_data
        self.path_to_data = path_to_data
        self.samples_per_class = 2


class TrainConfig:
    def __init__(self):
        self.log_interval = 10
        self.train_batch_size = 8
        self.test_batch_size = 4
        self.epochs = 10

