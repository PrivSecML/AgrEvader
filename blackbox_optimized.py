from organizer import *

#Set up the logger and the device
logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_blackbox_op_{}'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_AGR,
                                                                               TRAIN_EPOCH,
                                                                               MAX_EPOCH - TRAIN_EPOCH,COVER_FACTOR))
#set up random seed
org = Organizer()
org.set_random_seed()
#strat the greybox attack
org.federated_training_black_box_optimized(logger)