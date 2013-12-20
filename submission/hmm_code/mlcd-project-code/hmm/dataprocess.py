__author__ = 'arenduchintala'
import numpy as np
from sklearn.hmm import GaussianHMM
from train_states import get_states
import itertools
import re
import pdb

END = -1e3
START = 1e3
np.set_printoptions(precision=2, linewidth=120)


def savesplits(fname, list_of_patient_file_paths, splits):
    writer = open(fname, 'w')
    for idx, split in enumerate(splits):
        patient_file = list_of_patient_file_paths[idx]
        patient_prediction = split
        writer.write(patient_file + '\t' + patient_prediction + '\n')
    writer.flush()
    writer.close()


def check_start_stop_index(start_stop_seq, best_seq, sirs_feats):
    #print best_seq[start_stop_idx]
    st = best_seq[start_stop_seq[0]]
    sp = best_seq[start_stop_seq[1]]
    st_obs = sirs_feats[start_stop_idx[0]]
    sp_obs = sirs_feats[start_stop_idx[1]]
    for i in range(0, len(start_stop_idx), 2):
        #print start_stop_idx[i], start_stop_idx[i + 1]
        assert st == best_seq[start_stop_idx[i]]
        assert sp == best_seq[start_stop_idx[i + 1]]
        assert np.array_equal(st_obs, sirs_feats[start_stop_idx[i]])
        assert np.array_equal(sp_obs, sirs_feats[start_stop_idx[i + 1]])
    return st, sp


def string_patient_feats(train_map, condition):
    patient_files = []
    patient_feats = []
    start_stop_idx = []
    for patient_condition, file_path, incident_time in train_map:
        if patient_condition == condition:
            feats = np.loadtxt(file_path)
            t, last_index = overlapped_samples(feats, incident_reported_time=int(incident_time), overlap=5, window=10)
            start_stop_idx.append(np.shape(t)[0] + 1)
            start_stop_idx.append(1)
            patient_feats.append(t)
            patient_files.append(file_path)
        else:
            pass
    start_stop_idx = np.cumsum(np.array(start_stop_idx))
    start_stop_idx = list(start_stop_idx.astype(int))
    start_stop_idx.pop()  # there is no start at the end
    start_stop_idx.insert(0, 0)
    return patient_feats, start_stop_idx, patient_files


def get_tmat_smat_with_start_and_end(number_of_pre_states):
    """
    makes the transition matrix and start probabilities for the hmm (initial values)
    """
    #  transition pre states
    x = np.array([])
    xs = float(1.0 / number_of_pre_states) * np.ones(number_of_pre_states)
    x = np.concatenate((x, np.array([0.0]), xs, np.array([0.0, 0.0, 0.0])))

    for i in range(number_of_pre_states):
        xpre = (1.0 / (number_of_pre_states + 1.0)) * np.ones(number_of_pre_states + 1)
        x = np.concatenate((x, np.array([0.0]), xpre, np.array([0.0, 0.0]))) # one 0 for post, one 0 for END


    # transition onset states
    xon = np.zeros(number_of_pre_states)
    x = np.concatenate(
        (x, np.array([0.0]), xon, np.array([0.5, 0.5, 0.0]))) # from a onset state it can stay in onset or go to post, last 0 is for END

    # transition post states
    xpo = np.zeros(number_of_pre_states + 1) # one zero for post - onset (not possible)
    x = np.concatenate((x, np.array([0.0]), xpo, np.array([0.1, 0.9])))  # 0.1 to stay in post, 0.9 to END

    #transition end state
    xe = np.zeros(number_of_pre_states + 2)
    x = np.concatenate((x, np.array([1.0]), xe, np.array([0.0])))
    t_prob = np.reshape(x, (number_of_pre_states + 4, number_of_pre_states + 4))
    start_prob = np.zeros(number_of_pre_states + 4)
    start_prob[0] = 1.0
    return t_prob, start_prob


def stack_patient_feats(patient_feats):
    """
    takes a python list of overlapped_samples (feats for a patient)
    and stacks them which START and END states between each overlapped_sample
    """
    window = np.shape(patient_feats[0])[1]  # should be the number of time stamps in one observation
    X = np.array([])
    for idx, p_feat in enumerate(patient_feats):
        #stick a patient feat in here
        if idx == 0:
            X = START * np.ones(window)
        else:
            X = np.vstack((X, START * np.ones(window)))
        X = np.vstack((X, p_feat))
        X = np.vstack((X, END * np.ones(window)))  # slap on a complementary end state

    return X


def overlapped_samples(feats, incident_reported_time, overlap, window, column_num=0):
    if len(np.shape(feats)) != 2:
        return None, None
    column_of_interest = feats[:, column_num]
    if len(column_of_interest) < incident_reported_time:
        return None, None

    data_of_interest = column_of_interest[0:incident_reported_time]

    i = 0
    until = incident_reported_time # len(data_of_interest)
    while (i + window) < until:
        x = data_of_interest[i:i + window]
        if i == 0:
            X = x
        else:
            X = np.vstack((X, x))
        i += (window - overlap)  # amount to shift is window - overlap amount

    return X, i


def split_best_seq(best_seq, start_sym, end_sym):
    best_seq_str = [str(b) for b in best_seq]
    best_seq_str = ','.join(best_seq_str)
    split_regex = re.compile('(?<=%s),(?=%s)' % (end_sym, start_sym))
    splits = re.split(split_regex, best_seq_str)
    return splits


if __name__ == "__main__":
    root = '../../lowres_features/'
    all_conditions = ['sirs'] # ['sevsep', 'sepshock']

    all_num_states = [8, 15, 20]
    all_num_iterations = [2, 5, 10, 15]
    for condition, n_states, n_iter in itertools.product(all_conditions, all_num_states, all_num_iterations):
        print 'training...', condition, n_states, n_iter
        feature = 'deviation'

        train_map = open(root + 'trainset.recs.updated.lowres.cleaned', 'r').readlines()
        train_map = [(line.split('/')[3], line.split('\t')[0], line.split('\t')[1]) for line in train_map]

        list_of_sirs_patients, start_stop_idx, list_of_patient_file_paths = string_patient_feats(train_map, condition)
        sirs_feats = stack_patient_feats(list_of_sirs_patients)
        print np.shape(sirs_feats)
        means, covs = get_states(n_states - 4, condition, feature, end=True, start=True)
        print means
        print covs
        tmat, smat = get_tmat_smat_with_start_and_end(n_states - 4)
        print tmat, smat
        model = GaussianHMM(n_components=n_states, covariance_type="diag", startprob=smat, transmat=tmat, n_iter=n_iter, init_params='mc')
        model.means_ = means
        model.covars_ = covs
        initial_ll, initial_best_seq = model.decode(sirs_feats)
        print 'initial log-likelihood,', initial_ll
        model.fit([sirs_feats])
        final_ll, final_best_seq = model.decode(sirs_feats)
        print 'final   log-likelihood,', final_ll
        st, sp = check_start_stop_index(start_stop_idx, final_best_seq, sirs_feats)
        splits = split_best_seq(final_best_seq, st, sp)
        print len(list_of_sirs_patients), len(splits), start_stop_idx
        assert len(list_of_patient_file_paths) == len(splits)
        #save all the files that have been generated by training
        mean_name = root + '/trained/' + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-mean.txt'
        cov_name = root + '/trained/' + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-cov.txt'
        tmat_name = root + '/trained/' + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-transtion.txt'
        pred_name = root + '/trained/' + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-prediction.txt'

        savesplits(pred_name, list_of_patient_file_paths, splits)
        np.savetxt(mean_name, model.means_, fmt='%.4f')
        final_covs = np.array([np.diag(model.covars_[0])])
        for i in range(1, model.n_components):
            final_covs = np.vstack((final_covs, [np.diag(model.covars_[i])]))
        np.savetxt(cov_name, final_covs, fmt='%.4f')
        np.savetxt(tmat_name, model.transmat_, fmt='%.2f')
        #print 'final means:\n', np.around(model.means_, 2)
        #print 'final tmat:\n', np.around(model.transmat_, 2)
