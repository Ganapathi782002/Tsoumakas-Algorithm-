import numpy as np
import pandas as pd
import pickle
import os
import re

# Specifying the path where our datasets are stored
DATASET_PATH = "C:/Users/admin/Desktop/ProjectWork2/data"

# Function to load a dataset based on its name
def load_dataset(dataset_name, debug=True):
    if dataset_name in ["mediamill","bibtex"]:
        return load_small_dataset(dataset_name, debug)
    else:
        return load_large_dataset(dataset_name, debug)


# Function to load a large-scale dataset
def load_large_dataset(dataset_name, debug=True):
    smalls = dataset_name.lower()
    train_path = os.path.join(DATASET_PATH, smalls, f"{smalls}_train.txt")
    test_path = os.path.join(DATASET_PATH, smalls, f"{smalls}_test.txt")
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
    trsplit_path = os.path.join(DATASET_PATH, firstcap, f"{dataset_name}_trSplit.txt")
    tstsplit_path = os.path.join(DATASET_PATH, firstcap, f"{dataset_name}_tstSplit.txt")

    if debug:
        print("Loading datasets")
        print(data_path)
        print(trsplit_path)
        print(tstsplit_path)
    trsplits, tst_splits = read_split_files(trsplit_path, tstsplit_path, debug)
    return read_data_file(data_path, debug), trsplits, tst_splits


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


# Function to read split files into Pandas DataFrames
def read_split_files(trsplit_path, tstsplit_path, debug):
    trsplits = pd.read_csv(trsplit_path, header=None, delim_whitespace=True)
    tst_splits = pd.read_csv(tstsplit_path, header=None, delim_whitespace=True)
    trsplits = trsplits - 1
    tst_splits = tst_splits - 1
    num_splits = len(trsplits.columns)
    assert len(tst_splits.columns) == num_splits
    if debug:
        print("Number of splits :", num_splits)
    return trsplits, tst_splits


# Function to get a specific split of a small dataset
def get_small_dataset_split(dataset, trn_splits, tst_splits, split_num):
    assert split_num < len(trn_splits)
    trn_data = dataset.iloc[trn_splits[split_num].to_numpy()]
    tst_data = dataset.iloc[tst_splits[split_num].to_numpy()]
    trn_data = trn_data.reset_index(drop=True)
    tst_data = tst_data.reset_index(drop=True)
    return trn_data, tst_data


# Function to get numpy arrays from a DataFrame
def get_arrays(data):
    y_mat = np.vstack(data["labels_binary"].to_numpy())
    x_mat = np.vstack(data["features"].to_numpy())
    return x_mat, y_mat


# Function to get a validation split
def get_validation_split(x_mat, y_mat, filename, valsplit=0.1):
    if os.path.exists(filename):
        with open(filename, "rb") as fi:
            data_dict = pickle.load(fi)
        trn_idcs = data_dict["trn"]
        val_idcs = data_dict["val"]
    else:
        num_val = int(x_mat.shape[0] * valsplit)
        perm = np.random.permutation(x_mat.shape[0])
        trn_idcs = perm[num_val:]
        val_idcs = perm[:num_val]
        with open(filename, "wb") as fi:
            data_dict = {"trn": trn_idcs, "val": val_idcs}
            pickle.dump(data_dict, fi)
    x_val = x_mat[val_idcs, :]
    y_val = y_mat[val_idcs, :]
    x_trn = x_mat[trn_idcs, :]
    y_trn = y_mat[trn_idcs, :]
    return x_trn, y_trn, x_val, y_val

if __name__ == "__main__":
    dataset_name = "bibtex"  
    result = load_dataset(dataset_name, debug=True)

    if dataset_name in ["mediamill", "bibtex"]:
        full_dataset, trn_splits, tst_splits = result
        trn_data = full_dataset  # full_dataset already contains trn_data
        tst_data = None  # You don't have tst_data in the result of load_dataset for small datasets

        print("\nTraining Dataset:")
        print(trn_data.head())

        # Iterate over all splits
        for split_num in range(len(trn_splits.columns)):
            trn_data_split, tst_data_split = get_small_dataset_split(trn_data, trn_splits, tst_splits, split_num)

            print(f"\nTraining Split {split_num + 1}:")
            print(trn_data_split[["features", "labels_list"]].head())

            print(f"\nTesting Split {split_num + 1}:")
            print(tst_data_split[["features", "labels_list"]].head())
    else:
        print("Unsupported dataset name.")
