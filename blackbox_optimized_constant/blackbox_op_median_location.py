from datetime import datetime
import torch

# General constants
PYTORCH_INIT = "PyTorch" #initialisation for pytorch`
now = datetime.now() #Current date and time
TIME_STAMP = now.strftime("%Y_%m_%d_%H") #Time stamp for logging
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Device for training, you can use cpu or gpu

# General hyper parameters for training
MAX_EPOCH = 1000 #Maximum number of epochs
TRAIN_EPOCH = 30 #Number of epochs for training
BATCH_SIZE = 64 #Batch size for training
RESERVED_SAMPLE = 300 #Number of sample for covering attack
INIT_MODE = PYTORCH_INIT # Initialisation mode
BATCH_TRAINING = True #Batch training or not

# Data set related
# You can explore more datasets here
LOCATION30 = "Location30" #Name of the dataset
LOCATION30_PATH = "./datasets-master/bangkok" #Path to the dataset
DEFAULT_SET = LOCATION30 #Default dataset

LABEL_COL = 0 #Label column
LABEL_SIZE = 30 #Number of classes
TRAIN_TEST_RATIO = (0.5, 0.5) #Ratio of training and testing data

# Robust AGR
MEDIAN = "Median" #Robust AGR -Median
FANG = "Fang" #Robust AGR -Fang
NONE = "None" #Average
DEFAULT_AGR = MEDIAN #Set default Robust AGR

# Federated learning parameters
NUMBER_OF_PARTICIPANTS = 5 #Number of participants
PARAMETER_EXCHANGE_RATE = 1 #Parameter exchange rate
PARAMETER_SAMPLE_THRESHOLD = 1 #Parameter sample threshold
GRADIENT_EXCHANGE_RATE = 1 #Gradient exchange rate
GRADIENT_SAMPLE_THRESHOLD = 1 #Gradient sample threshold


# Attacker related
NUMBER_OF_ADVERSARY = 1 #Number of adversaries
NUMBER_OF_ATTACK_SAMPLES = 300 #Number of attack samples
ASCENT_FACTOR = 1 #Ascent factor for Gradient Ascent
BLACK_BOX_MEMBER_RATE = 0.5 # Member sample rate for black box attack
FRACTION_OF_ASCENDING_SAMPLES = 1 #Fraction of ascending samples
COVER_FACTOR = 0.2 #Cover factor for covering attack
GREY_BOX_SHUFFLE_COPIES = 5# Attacker shuffle parameter for related for Greybox attack


# IO related
EXPERIMENTAL_DATA_DIRECTORY = "./output/"

# Random seed
GLOBAL_SEED = 999
