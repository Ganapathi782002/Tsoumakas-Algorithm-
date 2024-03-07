import numpy as np
import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import StratifiedShuffleSplit

# Specifying the path where our datasets are stored
DATASET_PATH = "C:/Users/admin/Desktop/ProjectWork2/data"

# Function to load a dataset based on its name
def load_dataset(dataset_name, debug=True):
    if dataset_name in ["mediamill", "bibtex"]:
        return load_small_dataset(dataset_name, debug)
    else:
        return load_large_dataset(dataset_name, debug)


# Function to load a large-scale dataset
def load_large_dataset(dataset_name, debug=True):
    smalls = dataset_name.lower()
    train_path = os.path.join(DATASET_PATH, smalls, f"{smalls}_train.txt")
    test_path = os.path.join(DATASET_PATH, smalls, f"{smalls}_test.txt")
    # print("Training dataset path: ",train_path)
    # print("Testing dataset path: ",test_path)
    trn_data = read_data_file(train_path, debug)
    tst_data = read_data_file(test_path, debug)
    return trn_data, tst_data


# Function to load a small-scale dataset
def load_small_dataset(dataset_name, debug=True):
    dataset_names = ["mediamill", "bibtex"]
    if dataset_name not in dataset_names:
        print("Invalid input")
        return
    firstcap = dataset_name.capitalize()

    # Specify path names based on DATASET_PATH
    data_path = os.path.join(DATASET_PATH, firstcap, f"{firstcap}_data.txt")

    if debug:
        print("Loading datasets")
        print(data_path)

    return read_data_file(data_path, debug)


# Function to read data from a file and create a DataFrame
def read_data_file(data_path, debug):
    header = None
    lines = []

    with open(data_path, "r") as daf:
        header = daf.readline()
        for l in daf:
            lines.append(l)

    num_points, num_features, num_labels = [int(x) for x in header.split()]

    if debug:
        print("## HEADER ##")
        print("#Point :", num_points, ", #Features :", num_features, ", #Labels :", num_labels)

    assert num_points == len(lines), "header num_points doesn't match with num_lines of file"

    all_points = []
    for i, line in enumerate(lines):
        point = {}
        match = re.search(r"\A((\d*,)*(\d+))\s", line)
        labstring = ""
        labels = []
        if match:
            labstring = match.groups()[0]
            labels = [int(l) for l in labstring.split(",")]

        featstring = line.replace(labstring, "", 1).strip()
        feats = featstring.split()
        feats = [f.split(":") for f in feats]
        feats = [(int(f[0]), f[1]) for f in feats]

        assert len(labels) <= num_labels
        if len(labels) > 0:
            assert max(labels) < num_labels
            assert min(labels) >= 0
        feat_idcs = [f[0] for f in feats]
        assert len(feats) <= num_features
        assert max(feat_idcs) < num_features
        assert min(feat_idcs) >= 0

        point["labels"] = labels
        point["features"] = feats
        all_points.append(point)

    assert len(all_points) == num_points

    x_mat = np.zeros((num_points, num_features), dtype=float)
    for i, p in enumerate(all_points):
        for f_idx, f_val in p["features"]:
            x_mat[i][f_idx] = f_val

    y_mat = np.zeros((num_points, num_labels), dtype=int)
    for i, p in enumerate(all_points):
        for l in p["labels"]:
            y_mat[i][l] = 1

    full_dataset = pd.DataFrame(
        {"features": [x_mat[i, :] for i in range(0, num_points)],
         "labels_binary": [y_mat[i, :] for i in range(0, num_points)],
         "labels_list": [all_points[i]["labels"] for i in range(0, num_points)]}
    )

    return full_dataset


# Function to split the dataset into training and testing using stratified sampling
def stratified_split(full_dataset):
    print("** USING STRATIFIED SAMPLING **")
    y_labels = full_dataset["labels_list"].apply(lambda x: ",".join([str(l) for l in x]))
    label_counts = y_labels.value_counts()

    # Exclude classes with less than 2 samples
    valid_classes = label_counts[label_counts >= 2].index

    # Filter the dataset to include only samples from valid classes
    full_dataset_filtered = full_dataset[y_labels.isin(valid_classes)]

    y_labels_filtered = full_dataset_filtered["labels_list"].apply(lambda x: ",".join([str(l) for l in x]))
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(full_dataset_filtered, y_labels_filtered):
        train_set = full_dataset_filtered.iloc[train_index]
        test_set = full_dataset_filtered.iloc[test_index]
    return train_set, test_set


if __name__ == "__main__":
    dataset_name = "bibtex"
    full_dataset = load_dataset(dataset_name, debug=True)

    if dataset_name in ["mediamill", "bibtex"]:
        # Split the dataset using stratified sampling
        train_set, test_set = stratified_split(full_dataset)

        print("\nTraining Dataset:")
        print(train_set.head())

        print("\nTesting Dataset:")
        print(test_set.head())
    else:
        print("Unsupported dataset name.")
