import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import sys
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.data_loader import EEGDataSet
from utils.test import test
from utils.parse_config import parse_config
from architecture.model_standard_EEGNet import EEGNet
from architecture.model_standard_IncpetionEEG import inceptionEEGNet
from architecture.model_standard_EEGSym import EEGSym
from architecture.model_standard_ShallowFBCSPNet import ShallowFBCSPNetIRT
from architecture.model_standard_Deep import DeepNetIRT


# Read parameters from the configuration file
config_file = 'config.ini'
config_dict = parse_config(config_file)

# Set parameters
print(config_dict['log_id'])
DL_file = config_dict['dataset_name']
batch_size = config_dict['batch_size']
n_epoch = config_dict['n_epoch']
cuda = config_dict['is_cuda']
EEG_data_root = config_dict['dataset_root']
lr = config_dict['lr']
model_cache = config_dict['model_cache']
model_root = config_dict['model_root']
is_debug = config_dict['is_debug']

# Fix random seed for reproducibility
seed = config_dict['random_seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Create the EEG classification model
my_net = EEGNet(num_channel=config_dict['num_channel'], num_class=config_dict['num_class'],
                len_window=config_dict['len_window'])


# Load the training and evaluation data
train_DL = os.path.join(DL_file, "MI_train0")
eval_DL = os.path.join(DL_file, "MI_eval0")
print(train_DL)
trainDataset = EEGDataSet(
    data_root=EEG_data_root,
    data_list=train_DL,
    num_channel=config_dict['num_channel'],
    len_data=config_dict['len_window']
)

trainDataloader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=config_dict['num_workers']
)

eval_dataset = EEGDataSet(
    data_root=EEG_data_root,
    data_list=eval_DL,
    num_channel=config_dict['num_channel'],
    len_data=config_dict['len_window']
)

eval_dataloader = torch.utils.data.DataLoader(
    dataset=eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=config_dict['num_workers']
)

# Set up the optimizer
optimizer = optim.AdamW(my_net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)

# Set up the loss function
loss_function = torch.nn.NLLLoss()
log_softmax = torch.nn.LogSoftmax(dim=1)
if cuda:
    my_net = my_net.cuda()
    loss_function = loss_function.cuda()
    log_softmax = log_softmax.cuda()
for p in my_net.parameters():
    p.requires_grad = True

# Initialize variables for recording best metrics
BestAUC = 0.0
BestACC = 0.0
BestF1Score = 0.0
best_index_AUC = 0
best_index_ACC = 0
best_index_F1 = 0

# Start the training process
for epoch in range(n_epoch):
    len_dataloader = len(trainDataloader)
    data_iter = iter(trainDataloader)
    my_net = my_net.train()
    for i in range(len_dataloader):

        # Read data from the training data loader
        my_data = next(data_iter)
        eegs, subject, label = my_data
        my_net.zero_grad()
        if cuda:
            eegs = eegs.cuda()
            label = label.cuda()

        # Feed data into the network
        class_output = my_net(input_data=eegs)

        # Compute and backpropagate the loss
        err = loss_function(log_softmax(class_output), label)
        err.backward()
        # Update network parameters and the learning rate scheduler
        optimizer.step()
        if is_debug:
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err: %f' \
                             % (epoch, i + 1, len_dataloader, err.data.cpu().numpy(),
                                ))
            sys.stdout.flush()
    scheduler.step()

    # Test the model on the evaluation dataset at the current epoch
    my_net = my_net.eval()

    len_eval_dataloader = len(eval_dataloader)
    val_dataloader_iter = iter(eval_dataloader)

    prob_all = []
    label_all = []
    test_count = 0
    while test_count < len_eval_dataloader:

        # Test the model using the evaluation data
        data_target = next(val_dataloader_iter)
        e_eeg, e_subject, e_label = data_target

        if cuda:
            e_eeg = e_eeg.cuda()
            e_label = e_label.cuda()

        prob = my_net(e_eeg)
        prob = prob.detach().cpu().numpy()
        prob_all.extend(np.argmax(prob, axis=1))
        label_all.extend(e_label.cpu().numpy())
        test_count += 1

    # Compute evaluation metrics
    re_AUC = roc_auc_score(label_all, prob_all)
    re_ACC = accuracy_score(label_all, prob_all)
    re_F1 = f1_score(label_all, prob_all)
    if is_debug:
        print("")
    if re_AUC > BestAUC:
        BestAUC = re_AUC
        best_index_AUC = epoch
        torch.save(my_net, os.path.join(model_root, model_cache + "_BestAUCModel"))
        if is_debug:
            print("Best AUC:{0} at {1}".format(re_AUC, best_index_AUC))
    if re_ACC > BestACC:
        BestACC = re_ACC
        best_index_ACC = epoch
        torch.save(my_net, os.path.join(model_root, model_cache + "_BestACCModel"))
        if is_debug:
            print("Best ACC:{0} at {1}".format(re_ACC, best_index_ACC))
    if re_F1 > BestF1Score:
        BestF1Score = re_F1
        best_index_F1 = epoch
        torch.save(my_net, os.path.join(model_root, model_cache + "_BestF1Model"))
        if is_debug:
            print("Best F1:{0} at {1}".format(re_F1, best_index_F1))

# Test the best model on the validation dataset
test_AUC, test_ACC, test_F1 = test(config_dict)
record = np.zeros([2, 3])
print('============ Final Summary ============= \n')
print("Eval Dataset:")
print("Best AUC")
print(BestAUC)
record[0, 0] = BestAUC
print("Best ACC")
print(BestACC)
record[0, 1] = BestACC
print("Best F1")
print(BestF1Score)
record[0, 2] = BestF1Score
print("Test Dataset:")
print("Best model's AUC")
print(test_AUC)
record[1, 0] = test_AUC
print("Best model's ACC")
print(test_ACC)
record[1, 1] = test_ACC
print("Best model's F1")
print(test_F1)
record[1, 2] = test_F1
np.save(os.path.join(config_dict['record_root'], config_dict['record_name']), record)
