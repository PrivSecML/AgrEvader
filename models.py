import logging
import os
import sys
from aggregator import *
from data_reader import DataReader



def make_logger(name, save_dir, save_filename):
    """
    Make a logger to record the training process
    :param name: logger name
    :param save_dir: the directory to save the log file
    :param save_filename: the filename to save the log file
    :return: logger
    """
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=DATE_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def select_by_threshold(to_share: torch.Tensor, select_fraction: float, select_threshold: float = 1):
    """
    Apply the privacy-preserving method following selection-by-threshold approach
    :param to_share: the tensor to share
    :param select_fraction: the fraction of the tensor to share
    :param select_threshold: the threshold to select the tensor
    :return: the shared tensor and the indices of the selected tensor
    """
    threshold_count = round(to_share.size(0) * select_threshold)
    selection_count = round(to_share.size(0) * select_fraction)
    indices = to_share.topk(threshold_count).indices
    perm = torch.randperm(threshold_count).to(DEVICE)
    indices = indices[perm[:selection_count]]
    rei = torch.zeros(to_share.size()).to(DEVICE)
    rei[indices] = to_share[indices].to(DEVICE)
    to_share = rei.to(DEVICE)
    return to_share, indices


class ModelLocation30(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelLocation30, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(446, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out

#

class TargetModel:
    """
    The model to attack against, the target for attacking
    """

    def __init__(self, data_reader: DataReader, participant_index=0, model=DEFAULT_SET):
        # initialize the model
        if model == LOCATION30:
            self.model = ModelLocation30()
        else:
            raise NotImplementedError("Model not supported")
        self.model = self.model.to(DEVICE)

        # initialize the data
        self.test_set = None
        self.train_set = None
        self.data_reader = data_reader
        self.participant_index = participant_index
        self.load_data()

        # initialize the loss function and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Initialize recorder
        self.train_loss = -1
        self.train_acc = -1


    def load_data(self):
        """
        Load batch indices from the data reader
        :return: None
        """
        self.train_set = self.data_reader.get_train_set(self.participant_index).to(DEVICE)
        self.test_set = self.data_reader.get_test_set(self.participant_index).to(DEVICE)



    def normal_epoch(self, print_progress=False, by_batch=BATCH_TRAINING):
        """
        Train a normal epoch with the given dataset
        :param print_progress: if print the training progress or not
        :param by_batch: True to train by batch, False otherwise
        :return: the training accuracy and the training loss value
        """
        train_loss = 0
        train_acc = 0
        batch_counter = 0
        if by_batch:
            for batch_indices in self.train_set:
                batch_counter += 1
                if print_progress and batch_counter % 100 == 0:
                    print("Currently training for batch {}, overall {} batches"
                          .format(batch_counter, self.train_set.size(0)))
                batch_x, batch_y = self.data_reader.get_batch(batch_indices)
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                out = self.model(batch_x).to(DEVICE)
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                batch_acc = (prediction == batch_y).sum()
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
        self.train_acc = train_acc / (self.train_set.flatten().size(0))
        self.train_loss = train_loss / (self.train_set.flatten().size(0))
        if print_progress:
            print("Epoch complete for participant {}, train acc = {}, train loss = {}"
                  .format(self.participant_index, train_acc, train_loss))
        return self.train_loss, self.train_acc

    def test_outcome(self, by_batch=BATCH_TRAINING):
        """
        Test through the test set to get loss value and accuracy
        :param by_batch: True to test by batch, False otherwise
        :return: the test accuracy and test loss value
        """
        test_loss = 0
        test_acc = 0
        if by_batch:
            for batch_indices in self.test_set:
                batch_x, batch_y = self.data_reader.get_batch(batch_indices)
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                # print(batch_x)
                with torch.no_grad():
                    out = self.model(batch_x).to(DEVICE)
                    batch_loss = self.loss_function(out, batch_y).to(DEVICE)
                    test_loss += batch_loss.item()
                    prediction = torch.max(out, 1).indices.to(DEVICE)
                    batch_acc = (prediction == batch_y).sum().to(DEVICE)
                    test_acc += batch_acc.item()
        test_acc = test_acc / (self.test_set.flatten().size(0))
        test_loss = test_loss / (self.test_set.flatten().size(0))
        return test_loss, test_acc

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to(DEVICE)
        with torch.no_grad():
            for parameter in self.model.parameters():
                out = torch.cat([out, parameter.flatten()]).to(DEVICE)
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.model.parameters():
            length = len(param.flatten())
            to_load = parameters[start_index: start_index + length].to(DEVICE)
            to_load = to_load.reshape(param.size()).to(DEVICE)
            with torch.no_grad():
                param.copy_(to_load).to(DEVICE)
            start_index += length

    def get_epoch_gradient(self, apply_gradient=True):
        """
        Get the gradient for the current epoch
        :param apply_gradient: if apply the gradient or not
        :return: the tensor contains the gradient
        """
        cache = self.get_flatten_parameters().to(DEVICE)
        self.normal_epoch()
        gradient = self.get_flatten_parameters() - cache.to(DEVICE)
        if not apply_gradient:
            self.load_parameters(cache)
        return gradient

    def init_parameters(self, mode=INIT_MODE):
        """
        Initialize the parameters according to given mode
        :param mode: the mode to init with
        :return: None
        """
        if mode == PYTORCH_INIT:
            return
        else:
            raise ValueError("Invalid initialization mode")

    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        :param gradient: the gradient to apply
        :return: the loss and accuracy of the test
        """
        cache = self.get_flatten_parameters()
        test_param = cache + gradient
        self.load_parameters(test_param)
        loss, acc = self.test_outcome()
        self.load_parameters(cache)
        return loss, acc




class FederatedModel(TargetModel):
    """
    Representing the class of federated learning members
    """
    def __init__(self, reader: DataReader, aggregator: Aggregator, participant_index=0):
        """
        Initialize the federated model
        :param reader: initialize the data reader
        :param aggregator: initialize the aggregator
        :param participant_index: the index of the participant
        """
        super(FederatedModel, self).__init__(reader, participant_index)
        self.aggregator = aggregator

    def init_global_model(self):
        """
        Initialize the current model as the global model
        :return: None
        """
        self.init_parameters()
        self.test_set = self.data_reader.test_set.to(DEVICE)
        self.train_set = None

    def init_participant(self, global_model: TargetModel, participant_index):
        """
        Initialize the current model as a participant
        :return: None
        """
        self.participant_index = participant_index
        self.load_parameters(global_model.get_flatten_parameters())
        self.load_data()

    def share_gradient(self):
        """
        Participants share gradient to the aggregator
        :return: None
        """
        gradient = self.get_epoch_gradient()
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices=indices, source=self.participant_index)
        return gradient

    def apply_gradient(self):
        """
        Global model applies the gradient
        :return: None
        """
        parameters = self.get_flatten_parameters()
        parameters += self.aggregator.get_outcome(reset=True)
        self.load_parameters(parameters)

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Participants collect parameters from the global model
        :param parameter: the parameters shared by the global model
        :return: None
        """
        to_load = self.get_flatten_parameters().to(DEVICE)
        parameter, indices = select_by_threshold(parameter, PARAMETER_EXCHANGE_RATE, PARAMETER_SAMPLE_THRESHOLD)
        to_load[indices] = parameter[indices]
        self.load_parameters(to_load)





class BlackBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to perform a black-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        """
        Initialize the black-box malicious participant
        :param reader: Reader to read the data
        :param aggregator: Global aggregator
        """
        super(BlackBoxMalicious, self).__init__(reader, aggregator)
        self.attack_samples, self.members, self.non_members = reader.get_black_box_batch()
        self.member_count = 0
        self.batch_x, self.batch_y = self.data_reader.get_batch(self.attack_samples)
        self.shuffled_y = self.shuffle_label(self.batch_y)
        for i in self.attack_samples:
            if i in reader.train_set:
                self.member_count += 1


    def shuffle_label(self, ground_truth: torch.Tensor):
        """
        Shuffle the labels of the given ground truth
        :param ground_truth: The ground truth to shuffled data
        :return: Shuffled labels
        """
        result = ground_truth[torch.randperm(ground_truth.size()[0])]
        for i in range(ground_truth.size()[0]):
            while result[i].eq(ground_truth[i]):
                result[i] = torch.randint(ground_truth.max(), (1,))
        return result

    def train(self):
        """
        Normal training process for the black-box malicious participant
        :return: Gradient of the current round
        """
        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)
        return gradient



    def blackbox_attack(self,cover_factor = 0,batch_size = BATCH_SIZE):
        """
        Optimized shuffle label attack
        :param cover_factor: Cover factor of the gradient of cover samples
        :param batch_size: The size of the batch
        :return: The malicious gradient covered by gradient of cover samples for current round
        """

        cache = self.get_flatten_parameters()
        out = self.model(self.batch_x)
        loss = self.loss_function(out, self.shuffled_y) # compute loss with shuffled labels
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        cover_samples = self.data_reader.reserve_set #cover samples
        i = 0
        while i * batch_size < len(cover_samples):
            batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        cover_gradient = self.get_flatten_parameters() - cache

        if RESERVED_SAMPLE != 0:
            gradient, indices = select_by_threshold(cover_gradient*cover_factor+gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD) # computed the malicious gradient
        else:
            gradient, indices = select_by_threshold(gradient,GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)

        return gradient

    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        :return: the number of true member, false member, true non-member, false non-member
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member

    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        :return: The accuracy rate of members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate.cpu().numpy() / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        :return: The accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate.cpu().numpy() / len(batch_y)


class GreyBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to collect data for a white-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        """
        Initialize the malicious participant
        :param reader: Reader for the data
        :param aggregator: Global aggregator
        """
        super(GreyBoxMalicious, self).__init__(reader, aggregator, 0)
        self.members = None
        self.non_members = None
        self.attack_samples = self.get_attack_sample()
        self.descending_samples = None
        self.shuffled_labels = {}
        self.shuffle_labels()
        self.global_gradient = torch.zeros(self.get_flatten_parameters().size())
        self.member_prediction = None


    def train(self, batch_size=BATCH_SIZE):
        """
        Start a white-box training
        :param batch_size: The batch size
        :return: The malicious gradient for normal training round
        """
        cache = self.get_flatten_parameters()
        descending_samples = self.data_reader.reserve_set
        i = 0
        while i * batch_size < len(descending_samples):
            batch_index = descending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)
        return gradient

    def greybox_attack(self, ascent_factor=ASCENT_FACTOR, batch_size=BATCH_SIZE,
                                  mislead=True, mislead_factor=1, cover_factor=1):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :param ascent_factor: The factor of the ascending gradient
        :param batch_size: The batch size
        :param mislead: Whether to perform misleading
        :param mislead_factor: The factor of the misleading gradient
        :param cover_factor: The factor of the descending gradient
        :return: malicious gradient generated
        """
        cache = self.get_flatten_parameters()
        self.descending_samples = self.data_reader.reserve_set
        # Perform gradient ascent for the attack samples
        i = 0
        while i * batch_size < len(self.attack_samples):
            batch_index = self.attack_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        asc_gradient = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        # Perform gradient descent for the rest of samples
        i = 0
        while i * batch_size < len(self.descending_samples):
            batch_index = self.descending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        desc_gradient = self.get_flatten_parameters() - cache
        if not mislead:
            return desc_gradient - asc_gradient * ascent_factor

        # mislead labels
        self.load_parameters(cache)
        mislead_gradients = []
        for k in range(len(self.shuffled_labels)):
            i = 0
            while i * batch_size < len(self.attack_samples):
                batch_index = self.attack_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                y = self.shuffled_labels[k][batch_index]
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            mislead_gradients.append(self.get_flatten_parameters() - cache)
            self.load_parameters(cache)

        # select the best misleading gradient
        selected_k = 0
        largest_gradient_diff = 0
        for k in range(len(mislead_gradients)):
            diff = (mislead_gradients[k] - asc_gradient).norm()
            if diff > largest_gradient_diff:
                largest_gradient_diff = diff
                selected_k = k

        gradient = cover_factor * desc_gradient - asc_gradient * 1 + mislead_factor * mislead_gradients[selected_k]
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)

        return gradient

    def get_attack_sample(self, attack_samples=NUMBER_OF_ATTACK_SAMPLES, member_rate=BLACK_BOX_MEMBER_RATE):
        """
        Randomly select a sample from the data set
        :param attack_samples: The number of attack samples
        :param member_rate: The rate of member samples
        :return: shuffled data of attacker samples
        """
        member_count = round(attack_samples * member_rate)
        non_member_count = attack_samples - member_count
        self.members = self.data_reader.train_set.flatten()[
            torch.randperm(len(self.data_reader.train_set.flatten()))[:member_count]]
        self.non_members = self.data_reader.test_set.flatten()[
            torch.randperm(len(self.data_reader.test_set.flatten()))[:non_member_count]]
        return torch.cat([self.members, self.non_members])[torch.randperm(attack_samples)]

    def shuffle_labels(self, iteration=GREY_BOX_SHUFFLE_COPIES):
        """
        Shuffle the labels in several random permutation, to be used as misleading labels
        it will repeat the given iteration times denote as k, k different copies will be saved
        :param iteration: The number of copies
        """
        max_label = torch.max(self.data_reader.labels).item()
        for i in range(iteration):
            shuffled = self.data_reader.labels[torch.randperm(len(self.data_reader.labels))]
            for j in torch.nonzero(shuffled == self.data_reader.labels):
                shuffled[j] = (shuffled[j] + torch.randint(max_label, [1]).item()) % max_label
            self.shuffled_labels[i] = shuffled


    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        :return: The accuracy rate of members
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        :return: The accuracy rate of non-members
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        :return: the number of true member, false member, true non-member, false non-member
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member