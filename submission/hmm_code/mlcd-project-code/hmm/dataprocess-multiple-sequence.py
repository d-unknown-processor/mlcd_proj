__author__ = 'arenduchintala'
import numpy as np
from sklearn.hmm import GaussianHMM
from nearPD import nearPD, nearPSD
import itertools
import re
import pdb

END = -1e3
START = 1e3
np.set_printoptions(precision=2, linewidth=120)


def make_copy(mean, cov, perturb=True):
    m = np.copy(mean)
    c = np.copy(cov)
    if perturb:
        mp = np.random.multivariate_normal(np.zeros(len(m)), 5 * np.identity(len(m)))
        m += mp
    return m, c


def get_initial_means_and_covs(root_path, temporal_stage, condition_name, feature, cov_type='diag'):
    """
    root_path path to folder were the stats are located
    temporal_stage folder name of temporal_stage of the incident e.g. pre,onset or post
    condition_name either sevsep, sepshock, or sirs
    feature name : this corresponds to the column of the feats file it can be deviation,skew-(left|right), (slow|fast)-dfa

    """
    constructed_path_mean = root_path + temporal_stage + '_stats/' + condition_name + '-' + feature + '-mean.txt'
    mean = np.loadtxt(constructed_path_mean)
    if cov_type == 'diag':
        constructed_path_cov = root_path + temporal_stage + '_stats/' + condition_name + '-' + feature + '-cov.txt'
        cov = np.loadtxt(constructed_path_cov)
    else:
        constructed_path_cov = root_path + temporal_stage + '_stats/' + condition_name + '-' + feature + '-cov-full.txt'
        cov = np.loadtxt(constructed_path_cov)

    return mean, cov


def get_initial_states(num_pre_states, condition, feature, end=False, start=False, cov_type='diag'):
    pre_mean, pre_cov = get_initial_means_and_covs('../../lowres_features/', 'pre', condition, feature, cov_type)
    means = np.array([pre_mean])
    covs = np.array([pre_cov])
    while np.shape(means)[0] < num_pre_states:
        m, c = make_copy(pre_mean, pre_cov)
        means = np.vstack((means, [m]))
        covs = np.vstack((covs, [c]))

    onset_mean, onset_cov = get_initial_means_and_covs('../../lowres_features/', 'onset', condition, feature, cov_type)
    means = np.vstack((means, [onset_mean]))
    covs = np.vstack((covs, [onset_cov]))

    post_mean, post_cov = get_initial_means_and_covs('../../lowres_features/', 'post', condition, feature, cov_type)
    means = np.vstack((means, [post_mean]))
    covs = np.vstack((covs, [post_cov]))

    if end:
        raise BaseException('should not use end states')
        end_mean = END * np.ones(len(pre_mean))
        if cov_type == 'diag':
            end_cov = EPS * np.ones(len(pre_mean))
        else:
            end_cov = EPS * np.identity(len(pre_mean))
        means = np.vstack((means, [end_mean]))
        covs = np.vstack((covs, [end_cov]))

    if start:
        raise BaseException('should not use start states')
        start_mean = START * np.ones(len(pre_mean))
        if cov_type == 'diag':
            start_cov = EPS * np.ones(len(pre_mean))
        else:
            start_cov = EPS * np.identity(len(pre_mean))
        means = np.vstack(([start_mean], means))
        covs = np.vstack(([start_cov], covs))
    return means, covs


def save_full(file_path, arr, num_states, window_size, ms):
    for i in range(num_states):
        if check_covar:
            try:
                np.linalg.cholesky(arr[i])
            except:
                print 'this matrx is not positive def:\n'
                print arr[i]
                print 'corresponding mean:\n'
                print ms[i]
                print 'eigs:\n'
                print np.linalg.eigvals(arr[i])
                arr[i] = nearPD(arr[i])
                print 'fixed...:\n'
                print arr[i]
                print np.linalg.eigvals(arr[i])
                np.linalg.cholesky(arr[i])
    reshaped_arr = np.reshape(arr, (num_states * window_size, window_size))
    np.savetxt(file_path, reshaped_arr, fmt='%.4f')
    print 'writing model...'


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


def string_patient_feats(train_map, condition, overlap, window):
    patient_files = []
    patient_feats = []
    start_stop_idx = []
    for patient_condition, file_path, incident_time in train_map:
        if patient_condition == condition:
            feats = np.loadtxt(file_path)
            t, last_index = overlapped_samples(feats, incident_reported_time=int(incident_time), overlap=overlap, window=window)
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


def get_tmat_and_smat(number_of_pre_states, end, start):
    """
    makes the transition matrix and start probabilities for the hmm (initial values)
    """
    #  transition pre states
    pre_trans = np.ones((number_of_pre_states, number_of_pre_states + 1)) * (1.0 / (1 + number_of_pre_states))
    pre_pads = np.zeros((number_of_pre_states, 1))

    onset_trans = np.concatenate((np.zeros(number_of_pre_states), np.array([0.5, 0.5])))
    post_trans = np.concatenate((np.zeros(number_of_pre_states + 1), np.array([1.0])))
    core = np.hstack((pre_trans, pre_pads))
    core = np.vstack((core, onset_trans, post_trans))

    if end:
        pass
    if start:
        pass
    '''
    x = np.array([])
    if start:
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
    '''
    start_prob = np.zeros(np.shape(core)[0])
    start_prob[0] = 1.0

    return core, start_prob


def list_patient_feats(patient_feats):
    """
    takes a python list of overlapped_samples (feats for a patient)
    and stacks them which START and END states between each overlapped_sample
    """
    '''
    window = np.shape(patient_feats[0])[1]  # should be the number of time stamps in one observation
    list_of_sequences = []
    for idx, p_feat in enumerate(patient_feats):
        #X = START * np.ones(window)  # slap on a complementary start state
        X = np.vstack((X, p_feat))  # stick a patient feat in here
        #X = np.vstack((X, END * np.ones(window)))  # slap on a complementary end state
        list_of_sequences.append(p_feat)
    '''
    return patient_feats


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


if __name__ == "__main__":
    '''
    a = np.array([[1, 1, -1], [1, 1, 1], [-1, 1, 1]])
    fixed = nearPSD(a)
    print fixed
    print np.linalg.eigh(fixed)
    fixed = nearPSD(a, 1e-3)
    print fixed
    print np.linalg.eigh(fixed)
    #print np.linalg.cholesky(fixed)
    pdb.set_trace()
    '''

    cov_type = 'full'
    window = 10
    overlap = 0
    check_covar = False
    #save_model_to = 'trained-full-cov-no-checks/'
    save_model_to = 'trained-' + cov_type + '-cov-no-overlap/'
    root = '../../lowres_features/'
    all_conditions = ['sirs', 'sevsep', 'sepshock']
    all_num_states = [15, 20, 25] #[4, 6, 8, 12]  #
    all_num_iterations = [1000]
    for condition, n_states, n_iter in itertools.product(all_conditions, all_num_states, all_num_iterations):
        print 'training...', condition, n_states, n_iter
        feature = 'deviation'
        pre_states = n_states - 2
        train_map = open(root + 'trainset.recs.updated.lowres.cleaned', 'r').readlines()
        train_map = [(line.split('/')[3], line.split('\t')[0], line.split('\t')[1]) for line in train_map]

        list_of_patient_feats, start_stop_idx, list_of_patient_file_paths = string_patient_feats(train_map, condition, overlap, window)
        #sirs_feats_stacked = stack_patient_feats(list_of_sirs_patients)
        feats_as_list = list_patient_feats(list_of_patient_feats)
        #print np.shape(sirs_feats_stacked)
        means, covs = get_initial_states(pre_states, condition, feature, end=False, start=False, cov_type=cov_type)
        print means
        print covs
        if cov_type == 'full':
            for i in range(n_states):
                print 'checking if initial covs are pos-definite'
                np.linalg.cholesky(covs[i])
                print np.linalg.eigvals(covs[i])
        tmat, smat = get_tmat_and_smat(pre_states, end=False, start=False)
        print tmat, smat
        model = GaussianHMM(n_components=n_states, n_iter=n_iter, covariance_type=cov_type, startprob=smat, transmat=tmat, init_params='mc')
        model.means_ = means
        model.covars_ = covs
        sum_inital_ll = 0.0
        sum_initial_score = 0.0
        sum_initial_map = 0.0
        remove_idx = []
        for idx, feat_from_list in enumerate(feats_as_list):
            if np.shape(feat_from_list)[0] > n_states:
                initial_ll, initial_best_seq = model.decode(feat_from_list)
                initial_map, initial_best_sep_map = model.decode(feat_from_list, algorithm='map')
                sum_initial_score += model.score(feat_from_list)
                sum_inital_ll += initial_ll
                sum_initial_map += initial_map
            else:
                remove_idx.append(idx)
                print 'too few samples in file', list_of_patient_file_paths[idx], np.shape(feat_from_list)
        print 'initial viterbi log-likelihood,', sum_inital_ll
        print 'initial score log-likelihood,', sum_initial_score
        print 'initial map log-likelihood', sum_initial_map
        remove_idx.sort()
        remove_idx.reverse()
        print 'removing...', remove_idx
        for r in remove_idx:
            del feats_as_list[r]
        model.fit(feats_as_list)
        sum_final_ll = 0.0
        sum_final_score = 0.0
        for feat_from_list in feats_as_list:
            print np.shape(feat_from_list)
            final_ll, final_best_seq = model.decode(feat_from_list)
            final_score = model.score(feat_from_list)
            sum_final_ll += final_ll
            sum_final_score += final_score
        print 'final viterbi log-likelihood,', sum_final_ll
        print 'final score log-likelihood,', sum_final_score

        #save all the files that have been generated by training

        mean_name = root + save_model_to + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-mean.txt'
        cov_name = root + save_model_to + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-cov.txt'
        tmat_name = root + save_model_to + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-transtion.txt'
        pred_name = root + save_model_to + condition + '/' + condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(
            model.n_iter) + '-iter-prediction.txt'

        #savesplits(pred_name, list_of_patient_file_paths, splits)
        np.savetxt(mean_name, model.means_, fmt='%.4f')
        if cov_type == 'diag':
            final_covs = np.array([np.diag(model.covars_[0])])
            for i in range(1, model.n_components):
                final_covs = np.vstack((final_covs, [np.diag(model.covars_[i])]))
            np.savetxt(cov_name, final_covs, fmt='%.4f')
        else:
            # if model.covars_ is full matrix, it is a 3d array we can save it using np.save
            save_full(cov_name, model.covars_, n_states, window, model.means_)
            #np.savetxt(cov_name, model.covars_, fmt='%.4f')
        np.savetxt(tmat_name, model.transmat_, fmt='%.2f')
        #print 'final means:\n', np.around(model.means_, 2)
        #print 'final tmat:\n', np.around(model.transmat_, 2)


