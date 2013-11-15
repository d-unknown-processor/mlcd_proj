__author__ = 'arenduchintala'
import os

"""
 finds missing files in the train, dev and test set files. (e.g. trainset.recs)
 removes them and makes a updated train,dev and set set (e.g. trainset.recs.updated)
"""

root = "../../"
sepshock_root = root + "features/sepshock/"
sevsep_root = root + "features/sevsep/"
sirs_root = root + "features/sirs/"
train_map = root + "trainset.recs"
test_map = root + "testset.recs"
dev_map = root + "devset.recs"
folder_map = {1: sirs_root, 3: sevsep_root, 4: sepshock_root}
data_maps = [train_map, test_map, dev_map]

for m in data_maps:
    updated = []
    for f in open(m).readlines():
        #"1" sirs
        #"4": septic shock
        #"3": severe sepsis
        toks = f.split()
        print toks
        file_look_up = folder_map[int(toks[2])] + toks[0] + '.feats'
        if os.path.exists(file_look_up):
            updated.append(file_look_up)
        else:
            print 'no file', file_look_up
        wrt = open(m + '.updated', 'w')
        wrt.write('\n'.join(updated))
        wrt.flush()
        wrt.close()