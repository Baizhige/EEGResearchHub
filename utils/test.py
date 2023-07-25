import os
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from .data_loader import EEGDataSet

def test(config_dict):
    """
    Test the model using the provided configuration.

    Args:
    - config_dict (dict): A dictionary containing the configuration parameters.

    Returns:
    - re_AUC (float): ROC AUC score.
    - re_acc (float): Accuracy score.
    - re_f1 (float): F1 score.
    """
    batch_size = config_dict['batch_size']
    cuda = config_dict['is_cuda']

    # Create the dataset for testing
    my_dataset = EEGDataSet(
        data_root=config_dict['dataset_root'],
        data_list=os.path.join(config_dict['dataset_name'], "MI_test0"),
        num_channel=config_dict['num_channel'],
        len_data=config_dict['len_window'],
    )

    # Create a data loader for the testing dataset
    dataloader = torch.utils.data.DataLoader(
        dataset=my_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config_dict['num_workers']
    )

    # Load the pre-trained model
    my_net = torch.load(os.path.join(
        config_dict['model_root'], config_dict['model_cache'] + "_BestAUCModel"
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)
    i = 0
    prob_all = []
    label_all = []

    # Iterate through the testing data
    while i < len_dataloader:
        # Get the data for testing
        data_target = next(data_target_iter)
        t_img, t_subject, t_label = data_target

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        # Test the model
        prob = my_net(t_img)
        prob = prob.detach().cpu().numpy()
        prob_all.extend(np.argmax(prob, axis=1))
        label_all.extend(t_label.cpu().numpy())
        i += 1

    # Calculate evaluation metrics
    re_acc = accuracy_score(label_all, prob_all)
    re_f1 = f1_score(label_all, prob_all)
    re_AUC = roc_auc_score(label_all, prob_all)
    return re_AUC, re_acc, re_f1
