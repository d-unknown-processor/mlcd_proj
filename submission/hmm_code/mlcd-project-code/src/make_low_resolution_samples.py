__author__ = 'arenduchintala'
"""
This file takes the .feats data and converts them to a lower time resolution
takes every 120th sample from .feats.
"""

import os
import pdb

root = "../../"
train_map = root + "trainset.recs.updated"
test_map = root + "testset.recs.updated"
dev_map = root + "devset.recs.updated"
all_lists = [dev_map, train_map, test_map]
for a_list in all_lists:
    lowres_list = []
    highres_file_list = open(a_list, 'r').readlines()
    for l in highres_file_list:
        [file_name, time_of_clinical_obs] = l.strip().split('\t')
        feat_file = open(file_name, 'r').readlines()
        sample_feats = [sample for idx, sample in enumerate(feat_file) if idx % 120 == 0]
        combined = ''.join(sample_feats)
        save_to = file_name.replace('features', 'lowres_features')
        lowres_list.append(save_to + '\t' + str(int(time_of_clinical_obs) / 120))
        #pdb.set_trace()
        writer = open(save_to, 'w')
        writer.write(combined)
        writer.flush()
        writer.close()
    listwriter = open(a_list + '.lowres', 'w')
    listwriter.write('\n'.join(lowres_list))
    listwriter.flush()
    listwriter.close()
    pdb.set_trace()

