import pandas as pd
from constants import *

class DataReader:
    """
    The class to read data set from the given file
    """
    def __init__(self, data_set=DEFAULT_SET, label_column=LABEL_COL, batch_size=BATCH_SIZE,
                reserved=0):
        """
        Load the data from the given data path
        :param path: the path of csv file to load data
        :param label_column: the column index of csv file to store the labels
        :param label_size: The number of overall classes in the given data set
        """
        # load the csv file
        if data_set == LOCATION30:
            path = LOCATION30_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)

        self.data = self.data.to(DEVICE)
        self.labels = self.labels.to(DEVICE)

        # if there is no reserved data samples defined, then set the reserved data samples to 0
        try:
            reserved = RESERVED_SAMPLE
        except NameError:
            reserved = 0

        # initialize the training and testing batches indices
        self.train_set = None
        self.test_set = None
        overall_size = self.labels.size(0)


        # divide data samples into batches, drop the last bit of data samples to make sure each batch is full sized
        overall_size -= reserved
        overall_size -= overall_size % batch_size
        rand_perm = torch.randperm(self.labels.size(0)).to(DEVICE)
        self.reserve_set = rand_perm[overall_size:]
        print("cover dataset size is {}".format(reserved))
        rand_perm = rand_perm[:overall_size].to(DEVICE)
        self.batch_indices = rand_perm.reshape((-1, batch_size)).to(DEVICE)
        self.train_test_split()

        print("Data set "+DEFAULT_SET+
              " has been loaded, overall {} records, batch size = {}, testing batches: {}, training batches: {}"
              .format(overall_size, batch_size, self.test_set.size(0), self.train_set.size(0)))


    def train_test_split(self, ratio=TRAIN_TEST_RATIO, batch_training=BATCH_TRAINING):
        """
        Split the data set into training set and test set according to the given ratio
        :param ratio: tuple (float, float) the ratio of train set and test set
        :param batch_training: True to train by batch, False will not
        :return: None
        """
        if batch_training:
            train_count = round(self.batch_indices.size(0) * ratio[0] / sum(ratio))
            self.train_set = self.batch_indices[:train_count].to(DEVICE)
            self.test_set = self.batch_indices[train_count:].to(DEVICE)
        else:
            train_count = round(self.data.size(0) * ratio[0] / sum(ratio))
            rand_perm = torch.randperm(self.data.size(0)).to(DEVICE)
            self.train_set = rand_perm[:train_count].to(DEVICE)
            self.test_set = rand_perm[train_count:].to(DEVICE)

    def get_train_set(self, participant_index=0):
        """
        Get the indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        batches_per_participant = self.train_set.size(0) // NUMBER_OF_PARTICIPANTS
        lower_bound = participant_index * batches_per_participant
        upper_bound = (participant_index + 1) * batches_per_participant
        return self.train_set[lower_bound: upper_bound]

    def get_test_set(self, participant_index=0):
        """
        Get the indices for each test batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each test batch
        """
        batches_per_participant = self.test_set.size(0) // NUMBER_OF_PARTICIPANTS
        lower_bound = participant_index * batches_per_participant
        upper_bound = (participant_index + 1) * batches_per_participant
        return self.test_set[lower_bound: upper_bound]


    def get_batch(self, batch_indices):
        """
        Get the batch of data according to given batch indices
        :param batch_indices: tensor[BATCH_SIZE], the indices of a particular batch
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        return self.data[batch_indices], self.labels[batch_indices]

    def get_black_box_batch(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        member_count = round(attack_batch_size * member_rate)
        non_member_count = attack_batch_size - member_count
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        non_member_indices = test_flatten[torch.randperm((len(test_flatten)))[:non_member_count]].to(DEVICE)
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        result = result[torch.randperm(len(result))].to(DEVICE)
        return result, member_indices, non_member_indices