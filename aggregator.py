from constants import *


class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """

    def __init__(self, sample_gradients: torch.Tensor, robust_mechanism=None):
        """
        Initiate the aggregator according to the tensor size of a given sample
        :param sample_gradients: The tensor size of a sample gradient
        :param robust_mechanism: the robust mechanism used to aggregate the gradients
        """
        self.sample_gradients = sample_gradients.to(DEVICE)
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(DEVICE)
        self.robust = RobustMechanism(robust_mechanism)

        # AGR related parameters
        self.agr_model = None #Global model Fang required

    def reset(self):
        """
        Reset the aggregator to 0 before next round of aggregation
        """
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size())
        self.agr_model_calculated = False

    def collect(self, gradient: torch.Tensor,source, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        :param gradient: The gradient calculated by a participant
        :param souce: The source of the gradient, used for AGR verification
        :param indices: the indices of the gradient, used for AGR verification

        """
        if sample_count is None:
            self.collected_gradients.append(gradient)
            if indices is not None:
                self.counter_by_indices[indices] += 1
            self.counter += 1
        else:
            self.collected_gradients.append(gradient * sample_count)
            if indices is not None:
                self.counter_by_indices[indices] += sample_count
            self.counter += sample_count

    def get_outcome(self, reset=False, by_indices=False):
        """
        Get the aggregated gradients and reset the aggregator if needed, apply robust aggregator mechanism if needed
        :param reset: Whether to reset the aggregator after getting the outcome
        :param by_indices: Whether to aggregate by indices
        """
        if by_indices:
            result = sum(self.collected_gradients) / self.counter_by_indices
        else:
            result = self.robust.getter(self.collected_gradients, malicious_user=NUMBER_OF_ADVERSARY)
        if reset:
            self.reset()
        return result

    def agr_model_acquire(self, model):
        """
        Make use of the given model for AGR verification
        :param model: The model used for AGR verification
        """
        self.agr_model = model
        self.robust.agr_model_acquire(model)



class RobustMechanism:
    """
    The robust aggregator applied in the aggregator
    """
    #predefined the list of participants indices and status in AGR
    appearence_list = [0,1,2,3,4,5]
    status_list = []

    def __init__(self, robust_mechanism):
        self.type = robust_mechanism
        if robust_mechanism is None:
            self.function = self.naive_average
        elif robust_mechanism == MEDIAN:
            self.function = self.median
        elif robust_mechanism == FANG:
            self.function = self.Fang_defense
        self.agr_model = None


    def agr_model_acquire(self, model: torch.nn.Module):
        """
        Acquire the model used for LRR and ERR verification in Fang Defense
        The model must have the same parameters as the global model
        :param model: The model used for LRR and ERR verification
        """
        self.agr_model = model

    def naive_average(self, input_gradients: torch.Tensor):
        """
        The naive aggregator
        :param input_gradients: The gradients collected from participants
        :return: The average of the gradients
        """
        return torch.mean(input_gradients, 0)

    def median(self, input_gradients: torch.Tensor,number_of_malicious):
        """
        The median AGR
        :param input_gradients: The gradients collected from participants
        :return: The median of the gradients
        """
        return torch.median(input_gradients, 0).values


    def Fang_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The LRR and ERR mechanism proposed in Fang defense
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        """
        # Get the baseline loss and accuracy without removing any of the inputs
        all_avg = torch.mean(input_gradients, 0)
        base_loss, base_acc = self.agr_model.test_gradients(all_avg)
        loss_impact = []
        err_impact = []
        # Get all the loss value and accuracy without ith input
        RobustMechanism.status_list = []
        for i in range(len(input_gradients)):
            avg_without_i = (sum(input_gradients[:i]) + sum(input_gradients[i+1:])) / (input_gradients.size(0) - 1)
            ith_loss, ith_acc = self.agr_model.test_gradients(avg_without_i)
            loss_impact.append(torch.tensor(base_loss - ith_loss))
            err_impact.append(torch.tensor(ith_acc - base_acc))
            RobustMechanism.status_list.append((i,ith_acc,ith_loss))
        loss_impact = torch.hstack(loss_impact)
        err_impact = torch.hstack(err_impact)
        loss_rank = torch.argsort(loss_impact, dim=-1)
        acc_rank = torch.argsort(err_impact, dim=-1)
        result = []
        for i in range(len(input_gradients)):
            if i in loss_rank[:-malicious_user] and i in acc_rank[:-malicious_user]:
                result.append(i)
        RobustMechanism.appearence_list = result
        return torch.mean(input_gradients[result], dim=0)

    def getter(self, gradients, malicious_user=NUMBER_OF_ADVERSARY):
        """
        The getter method applying the robust AGR
        :param gradients: The gradients collected from all participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after adding the malicious gradient
        """
        gradients = torch.vstack(gradients)
        return self.function(gradients, malicious_user)
