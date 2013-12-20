__author__ = 'arenduchintala'
"""
The gaussianHMM requires data to be in the following format:
x = np.array([[1,2,3][1,2,3][1,2,3],...]) etc
each row e.g. [1,2,3] represents one time sclice

In our data we have a long time series sample in each of the .feats files (considering only first column in that file)

This method takes 60 samples from that file, takes the first number (std. deviation) and makes a x matrix
x = np.array([[1,2,...60][1,2,...60][1,2,...,60],...])
ofcourse the line above just says 1,2..60 which means a 60dim vector
Overlap param says how much overlap there is between the first 60dim row and the second 60dim row.
0 overlap means entirely new batch of 60 samples.
"""

EPS = 0.0
import numpy as np


def overlapped_samples(feats, incident_reported_time, overlap=3, window=6, column_num=0):
    column_of_interest = feats[:, column_num]

    #column_of_interest = column_of_interest[0:20] #fake truncate
    #print 'col of interest:\n', column_of_interest

    data_of_interest = column_of_interest[0:incident_reported_time]

    #pad__with_zero = np.ones(len(column_of_interest) - incident_reported_time) * EPS
    #data_of_interest = np.concatenate((data_of_interest, pad__with_zero))

    #print 'formatted of interest:\n', formatted_column
    X = np.array([])
    i = 0
    until = incident_reported_time # len(data_of_interest)
    while (i + window) < until:
        x = data_of_interest[i:i + window]
        if i == 0:
            X = x
        else:
            X = np.vstack((X, x))
        i += (window - overlap)  # amount to shift is window - overlap amount

    X = np.vstack((X, EPS * np.ones(window)))  # slap on a complementary end state
    return X, i


#data = np.loadtxt('../../lowres_features/sepshock/s32293-2595-04-09-21-46.feats')
#t, last_index = overlapped_samples(data, incident_reported_time=1516, overlap=30, window=60)
#print np.shape(t), last_index


