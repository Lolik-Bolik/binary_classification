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
        self.model_name = 'dummy_model'
        self.log_interval = 2
        self.train_batch_size = 32
        self.test_batch_size = 16
        self.num_workers = 4
        self.epochs = 100
        self.datapath = 'faces'
        self.criterion = 'labelsmoothing'  #'labelsmoothing'

