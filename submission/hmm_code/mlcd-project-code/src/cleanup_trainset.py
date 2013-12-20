__author__ = 'arenduchintala'
"""
lot of files in the trainset.vec are shitty.
this script removes them.

"""
import numpy as np

root = '../../lowres_features/'
train = root + 'trainset.recs.updated.lowres'
dev = root + 'devset.recs.updated.lowres'
test = root + 'testset.recs.updated.lowres'
train_cleaned = root + 'trainset.recs.updated.lowres.cleaned'
test_cleaned = root + 'testset.recs.updated.lowres.cleaned'
dev_cleaned = root + 'devset.recs.updated.lowres.cleaned'

train_map = open(train, 'r').readlines()
test_map = open(test, 'r').readlines()
dev_map = open(dev, 'r').readlines()
pairs = [(train_map, train_cleaned), (dev_map, dev_cleaned), (test_map, test_cleaned)]

for a_map, a_clean in pairs:
    ok_map = []
    for t in a_map:
        a_file_path = t.split('\t')[0]
        sample_of_occurance = int(t.split('\t')[1])
        feats = np.loadtxt(a_file_path)
        time_issue = False
        same_issue = False
        if np.shape(feats)[0] < sample_of_occurance:
            print "TIME INCIDENT ISSUE IN:", t.strip()
            time_issue = True
        else:
            r = 1
            found_diff = False
            while not found_diff and r < sample_of_occurance:
                found_diff = found_diff or not np.array_equal(feats[r], feats[r - 1])
                r += 1
            if not found_diff:
                print 'THIS FILE IS ALL THE SAME', t.strip()
                same_issue = True
        if not (same_issue or time_issue):
            print 'FILE OK', t.strip()
            ok_map.append(t.strip())

    writer = open(a_clean, 'w')
    writer.write('\n'.join(ok_map))
    writer.flush()
    writer.close()

