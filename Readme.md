# AgrEvader

## Cite AgrEvader
If you use AgrEvader for research, please cite the following paper:

NEED CITATION


## Code structure
To run the experiments, please run *setting*_optimized.py, replace *setting* with the AgrEvader knowledge settings to experiment with.
Here, we list two types of optimized attack based on the AgrEvader, Gray-box and Black-box. List of runnable script files
* blackbox_optimized.py (Refer to  Sec 5.1 Black-box AgrEvader)
* greybox_optimized.py (Refer to  Sec 5.2 Gray-box AgrEvader)


Other files in the repository
* __constants.py__ Experiment constants, contains the default hyperparameter set for each experiment
* __data_reader.py__ It is used for read dataset, split dataset into training set and test set for each participant
* __aggregator.py__ Byzantine-robust aggregators, including
  * Fang [1] (Refer to Sec 2.1)
  * Median [2] (Refer to Sec 2.1)
  
* __models.py__ The models including target and attack model are wrapped in this file, and also including
  * Black-box AgrEvader Attack Algorithm (Refer to Sec 5.1)
  * Gray-box AgrEvader Attack Algorithm (Refer to Sec 5.2)
 
* __organizer.py__ It is used for organizing the experiment, including
  * Black-box AgrEvader FL Framework(Refer to Sec 5.1)
  * Gray-box AgrEvader FL Framework (Refer to Sec 5.2)

## Instructions for running the experiments
### 1. Set the experiment parameters
The experiment parameters are defined in __constants.py__. To reproduce the results in the paper, please set the parameters as corresponding to the experiment settings in the __experiment settings__ directory. 

### 2. Run the experiment
To run the experiment, please run *setting*_optimized.py, replace *setting* with the AgrEvader knowledge settings to experiment with. You can use command line to run the experiment, e.g. in a LINUX environment, to execute the *Black-box AgrEvader* experiment, please input the following command under the source code path

```python blackbox_optimized.py```

To execute the *Gray-box AgrEvader* experiment, please input the following command under the source code path

```python greybox_optimized.py```

### 3. Save the experiment results
After the experiment is finished, the experiment results will be saved in the directory defined in __constants.py__ . The experiment results include the AgrEvader attack log, the global model and local models log, and the FL training log.

## Understanding the output
The result of the experiment will be saved in the directory defined in __constants.py__ (default *output* directory), including
* AgrEvader log file (E.g. 2023_02_08_00Location30FangTrainEpoch30AttackEpoch970blackboxoptimized_attacker.csv): This file contains the AgrEvader attack log, including:
  * acc & precision & recall: attack model accuracy, precision and recall
  * pred_acc_member: predicted accuracy of the member
  * pred_acc_non_member: predicted accuracy of the non-member
  * true_member:  the number of true membership of the member samples
  * false_member: the number of false membership of the member samples
  * true_non_member: the number of true membership of the non-member samples
  * false_non_member: the number of false membership of the non-member samples
  

* Global model and local models log file (E.g.2023_02_08_00Location30FangTrainEpoch30AttackEpoch970blackboxoptimized_model.csv): This file contains the global model and local models log, including:
  * participant: the id of the participant or global model
  * test loss: the test loss of the participant or global model
  * test accuracy: the test accuracy of the participant or global model
  * train accuracy: the train accuracy of the participant or global model
  
* FL training log file (E.g.log_2023_02_08_00_Location30_Fang_TrainEpoch30_AttackEpoch970_blackbox_op_0.2.txt): This file contains the FL training log, including:
  * round: the round of the FL training
  * test loss: the test loss of the global model for current round
  * test accuracy: the test accuracy of the global model for current round
  * train accuracy: the train accuracy of the global model for current round
  * time: the current time of the FL training


## Requirements
Recommended to run with conda virtual environment
* Python 3.10.5
* PyTorch 1.13.0
* numpy 1.22.4
* pandas 1.4.2

## Reference
[1] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. 2020. Local model poisoning attacks to byzantine-robust federated learning. In 29th {USENIX} Security Symposium ( {USENIX } Security 20). 1605–1622.

[2] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. 2018. Byzantine-robust distributed learning: Towards optimal statistical rates. In Inter- 1246 national Conference on Machine Learning. PMLR, 5650–5659.