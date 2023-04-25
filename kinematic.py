
# Example evaluating of models trained by us for emotion recognition
# from kinematic dataset on randomly selected samples
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset

################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 16

example_datasets = r".\Example datasets\\"
all_datasets = os.listdir(example_datasets)
"""
#example_datasets#
Small datasets are prepared separately for each window size.
In addition, separate data sets were designed for the data's feature extracted and the raw position state of the data.
In other words, 12 small new data sets were prepared in total. Each dataset has 100 sample windows of each emotion.
In total, there are 700 files in each dataset.
"""

trained_CNN_DL_Models = r".\Trained DL Models\\"
all_model_files = os.listdir(trained_CNN_DL_Models)
"""
The "trained_CNN_DL_Models" file contains 24 models containing the trained models of 
the MobileNetV3 and RegnetY800mf algorithms, where 12 small datasets can be tested.
"""

for s_dataset in range(len(all_datasets)):
    file_path = example_datasets + all_datasets[s_dataset]
    # We read each small dataset.
    with open(file_path, "rb") as fp:
        selected_dataset = pickle.load(fp)

    # Both labels and input features are converted to tensors.

    # Here, the labels are compatible with the torch and categorically corrected.
    le = LabelEncoder()
    tensor_label = torch.Tensor(le.fit_transform(np.array(selected_dataset[:, -1:]).ravel())).type(torch.LongTensor)

    tensor_features = torch.Tensor(selected_dataset[:, :-1])  # input features also converted to tensor.

    dataset_kinematic = TensorDataset(tensor_features, tensor_label)

    # Since the randomly selected dataset here is purely for testing purposes, all of them are chosen for testing.
    test_size = len(dataset_kinematic)

    _, test_dataset = torch.utils.data.random_split(dataset_kinematic, [0, test_size])

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size, shuffle=True)

    # The aim is to automate the processes by selecting the appropriate sample dataset
    # from the entire model file to be tested.
    splitted_filename = all_datasets[s_dataset].split("_")
    model_list = []
    file_lst = []
    str1 = splitted_filename[1] + "_"
    str2 = splitted_filename[0]
    for path, subdirs, files in os.walk(trained_CNN_DL_Models):
        for file in files:
            if (str1 in str(file)) and (str2 in str(file)):
                model_list.append(trained_CNN_DL_Models + file)
                file_lst.append(file)

    for mdl_c in range(len(model_list)):

        # # Automatically LOAD associated model.
        model = torch.jit.load(model_list[mdl_c], map_location=torch.device(device))

        # Network enters evaluation mode
        model.eval()

        print("The process for the '{}' file begins::".format(file_lst[mdl_c]))

        y_pred = []
        y_true = []
        # iterate over test data

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)  # Feed Network
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        acc_result = accuracy_score(y_true, y_pred)

        print("Test success rate of '{}' file is: {}".format(file_lst[mdl_c], acc_result * 100))
    del dataset_kinematic, test_dataloader, tensor_label, tensor_features, model
