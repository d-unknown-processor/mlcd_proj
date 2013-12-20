__author__ = 'arenduchintala'

"""
Go through the examples of sepsis/sevsepsis/sepshock/sirs is that patient exists in the train set.
stack the examples.
Note: each example has 60 time slice each time slice contains 9 numbers.

std deviation, slow DFA, fast DFA, skew left, skew right, entropy[0,..,4]
"""

import numpy as np
import pdb
import logging

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG)

time_width = 10
#SURE = 'sure_'  # ='' SET THIS TO EMPTY STRING FOR normal examples i.e. +/- 30 samples from time of incident
types = ['pre_', 'post_', 'onset_'] #SET THIS TO 'initial_' for patient start examples
types = ['sanity_after_', 'sanity_before_']
for SURE in types:
    namemap = {0: '-deviation', 1: '-slow-dfa', 2: '-fast-dfa', 3: '-skew-left', 4: '-skew-right'}
    stacked_cols = {}
    mean_cols = {}
    cov_cols = {}

    root = "../../lowres_features/"
    train_map = root + "trainset.recs.updated.lowres.cleaned"
    train_set = open(train_map, 'r').readlines()
    for a_lowres_train in train_set:
        a_lowres_train = a_lowres_train.split('\t')[0] # drop the second part of the line with the sample number of incident
        incident = a_lowres_train.split('/')[3]
        a_lowres_train = a_lowres_train.replace('/lowres_features/', '/lowres_features/' + SURE + 'examples/')
        a_lowres_example = a_lowres_train.replace('.feats', '-' + str(time_width) + '.' + SURE + 'examples')
        #a_lowres_example contains 60 min of samples centered at the time of reporting of incident
        try:
            examples = np.loadtxt(a_lowres_example)
            if np.shape(examples) == (0,):
                #skip
                pass
            elif np.shape(examples) == (time_width, 9):
                #process
                for i in range(5):
                    #pdb.set_trace()
                    col = examples[:, i]
                    key = (incident, i)
                    if key not in stacked_cols:
                        #print 'initial col for', key
                        stacked_cols[key] = col
                    else:
                        stacked_cols[key] = np.vstack((stacked_cols[key], col))
                        #print 'stacked', np.shape(stacked_cols[key]), key

            else:
                #do not have 60 samples what TODO?
                pass
        except IOError, iox:
            logging.exception('IO-Error' + a_lowres_example + 'must be an empty file... SKIPPING')

            exit()


    #stacking is done now compute means and covs
    write_stats = root + SURE + "stats/"
    for key in stacked_cols:
        try:
            filename_mean = write_stats + key[0] + namemap[key[1]] + '-mean.txt'
            filename_cov = write_stats + key[0] + namemap[key[1]] + '-cov.txt'
            filename_cov_full = write_stats + key[0] + namemap[key[1]] + '-cov-full.txt'
            print 'filenames:', filename_cov, filename_mean
            print 'shape of stacked_cols', np.shape(stacked_cols[key])
            mean_cols[key] = np.mean(stacked_cols[key], axis=0)
            np.savetxt(filename_mean, mean_cols[key])
            print 'shape of mean', key, np.shape(mean_cols[key])
            cov_cols[key] = np.cov(stacked_cols[key].T)
            np.savetxt(filename_cov, np.diag(cov_cols[key]))
            np.savetxt(filename_cov_full, cov_cols[key])
        except:
            print 'failed', key
            raise BaseException('failed')