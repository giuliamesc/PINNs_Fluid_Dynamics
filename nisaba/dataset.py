import tensorflow as tf

class DataEmpty():
    def __init__(self):
        pass

    def initialize(self):
        pass

    def next_batch(self):
        pass

    @property
    def current_batch(self):
        return None

    def get_tensor_spec(self):
        return None

    def get_sample_element(self):
        return None

    def set_batch_size(self, batch_size):
        pass

    def set_epochs_per_batch(self, epochs_per_batch):
        pass

class DataCollection():
    def __init__(self, datasets):
        datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

        #TODO: check for duplicates
        self.datasets = {dataset.name : dataset for dataset in datasets}

        self.__initialized = False
        self.__iterating = False

    def initialize(self):
        for _, dataset in self.datasets.items():
            dataset.initialize()

        self.__initialized = True
        self.__iterating = False

    def next_batch(self):
        if not self.__initialized:
            raise Exception('The DataCollection is not initialized.')
        self.__iterating = True
        for _, dataset in self.datasets.items():
            dataset.next_batch()

        self.__current_batch = {dataset.name : dataset.current_batch for _, dataset in self.datasets.items()}
        return self.__current_batch

    @property
    def current_batch(self):
        if not self.__initialized:
            raise Exception('The DataCollection is not initialized.')
        if not self.__iterating:
            raise Exception('The DataCollection has not started iterating.')

        return self.__current_batch

    def get_tensor_spec(self):
        return {dataset.name : dataset.get_tensor_spec() for _, dataset in self.datasets.items()}

    def get_sample_element(self):
        return {dataset.name : dataset.get_sample_element() for _, dataset in self.datasets.items()}

    def set_batch_size(self, batch_size):
        if isinstance(batch_size, dict):
            for key in batch_size:
                self.datasets[key].set_batch_size(batch_size[key])
        else:
            for _, dataset in self.datasets.items():
                dataset.set_batch_size(batch_size)

    def set_epochs_per_batch(self, epochs_per_batch):
        if isinstance(epochs_per_batch, dict):
            for key in epochs_per_batch:
                self.datasets[key].set_epochs_per_batch(epochs_per_batch[key])
        else:
            for _, dataset in self.datasets.items():
                dataset.set_epochs_per_batch(epochs_per_batch)


class DataSet():

    def __init__(self, data, name = 'data', batch_size = None, epochs_per_batch = 1, shuffle = False):
        self.data = data
        self.name = name
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch
        self.shuffle = shuffle

        # n_samples is the length of the first axis
        data_detail = data
        while isinstance(data_detail, (list, tuple)):
            data_detail = data_detail[0]
        self.n_samples = data_detail.shape[0]

        self.tf_dataset = tf.data.Dataset.from_tensor_slices(data)

        self.__initialized = False
        self.__iterating = False

    def initialize(self):

        if self.batch_size is None:
            batch_size = self.n_samples
        else:
            batch_size = self.batch_size

        self.tf_dataset_batch = self.tf_dataset
        if self.shuffle:
            self.tf_dataset_batch = self.tf_dataset_batch.shuffle(buffer_size = self.n_samples)
        self.tf_dataset_batch = self.tf_dataset_batch.repeat().batch(batch_size)
        self.__epoch_counter = 0
        self.tf_dataset_batch= iter(self.tf_dataset_batch)

        self.__initialized = True
        self.__iterating = False

    def next_batch(self):
        if not self.__initialized:
            raise Exception('The dataset %s is not initialized.' % self.name)
        self.__iterating = True
        if self.__epoch_counter % self.epochs_per_batch == 0:
            self.__current_batch = next(self.tf_dataset_batch)
        self.__epoch_counter +=1
        return self.__current_batch

    @property
    def current_batch(self):
        if not self.__initialized:
            raise Exception('The dataset %s is not initialized.' % self.name)
        if not self.__iterating:
            raise Exception('The dataset %s has not started iterating.' % self.name)

        return self.__current_batch

    def get_tensor_spec(self):
        return self.tf_dataset.batch(1).element_spec

    def get_sample_element(self):
        return next(iter(self.tf_dataset.batch(1)))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epochs_per_batch(self, epochs_per_batch):
        self.epochs_per_batch = epochs_per_batch