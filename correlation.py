# -*- coding: utf-8 -*-
from read_data import normalize_data
from scipy.stats import pearsonr
lenth, dimensions = normalize_data.shape
# get_data

def get_pearson():
    pearson_dict = {}
    for i in range(dimensions):
        for j in range(i+1, dimensions):
            r = pearsonr(normalize_data[:,i], normalize_data[:,j])
            pearson_dict[(i, j)] = pearson_dict[(j, i)] = abs(r[0]) 
    return pearson_dict
pearson_dict = get_pearson()


if __name__ == "__main__":
    get_pearson()