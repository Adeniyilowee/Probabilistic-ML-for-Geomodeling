# Importing necessary libraries and functions
import numpy as np
import main


def read_data(data, w):
    # data is combination of different surfaces in 3d
    sub_sample = []
    sub_sample_flat = []
    n_layer = len(data)
    for i in range(n_layer):
        layer_samples = all_sample(data[i], w)
        flat = [item for all_sublist in data[i] for item in all_sublist]
        sub_sample.append(layer_samples)
        sub_sample_flat.append(flat)

    return main.Fit(sub_sample, sub_sample_flat, n_layer)        # , data, data_points_x, data_points_y


def all_sample(data, w):
    data = sample(data, w)
    data = np.rot90(data, 3).tolist()
    data = sample(data, w)
    data = np.rot90(data, 9).tolist()
    return data


def sample(data, w):
    t = []
    for i in range(len(data)):
        points = []
        total_lenght = len(data[i])
        subsample = data[i][0:total_lenght:w]
        for n in subsample:
            points.append(n)

        if points[-1] != data[i][-1]:
            points.append(data[i][-1])

        else:
            points = points
        t.append(points)
    return t
