__author__ = 'arenduchintala'
import os
import pdb
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
        #print toks
        #toks[0] = s00292-3050-10-10-19-46 e.g
        #toks[1] = 13946
        #toks[2] = 4
        #pdb.set_trace()
        file_look_up = folder_map[int(toks[2])] + toks[0] + '.feats'
        if os.path.exists(file_look_up):
            updated.append(str(file_look_up+'\t'+toks[1]))
        else:
            print 'no file', file_look_up
        wrt = open(m + '.updated', 'w')
        wrt.write('\n'.join(updated))
        wrt.flush()
        wrt.close()