__author__ = 'arenduchintala'
"""
this script reads in the inital values of state means and variances
it gets the overlapped data in the correct format from 'make_overlapped_obs.py'
"""
import numpy as np
from sklearn.hmm import GaussianHMM

EPS = 1e-10
END = -1e3
START = 1e3
np.set_printoptions(precision=2, linewidth=120)


def overlapped_samples(file_path, incident_reported_time, overlap=3, window=6, with_end=0, column_num=0):
    feats = np.loadtxt(file_path)
    if len(np.shape(feats)) != 2:
        return None, None
    column_of_interest = feats[:, column_num]
    if len(column_of_interest) < incident_reported_time:
        return None, None

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

    for _ in range(with_end):
        X = np.vstack((X, END * np.ones(window)))  # slap on a complementary end state
    return X, i


def get_means_and_vars_of_states(root_path, temporal_stage, condition_name, feature, cov_type='diag'):
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


def make_copy(mean, cov, perturb=True):
    m = np.copy(mean)
    c = np.copy(cov)
    if perturb:
        mp = np.random.multivariate_normal(np.zeros(len(m)), np.identity(len(m)))
        m += mp
    return m, c


def get_tmat_smat_with_end(number_of_pre_states):
    """
    makes the transition matrix and start probabilities for the hmm (initial values)
    """
    #  transition pre states
    x = np.array([])

    for i in range(number_of_pre_states):
        xpre = (1.0 / (number_of_pre_states + 1.0)) * np.ones(number_of_pre_states + 1)
        x = np.concatenate((x, xpre, np.array([0.0, 0.0]))) # one 0 for post, one 0 for END


    # transition onset states
    xon = np.zeros(number_of_pre_states)
    x = np.concatenate((x, xon, np.array([0.5, 0.5, 0.0]))) # from a onset state it can stay in onset or go to post, last 0 is for END

    # transition post states
    xpo = np.zeros(number_of_pre_states + 1) # one zero for post - onset (not possible)
    x = np.concatenate((x, xpo, np.array([0.1, 0.9])))  # 0.1 to stay in post, 0.9 to END

    #transition end state
    xe = np.zeros(number_of_pre_states + 2)
    x = np.concatenate((x, xe, np.array([1.0])))
    t_prob = np.reshape(x, (number_of_pre_states + 3, number_of_pre_states + 3))
    start_prob = np.zeros(number_of_pre_states + 3)
    start_prob[0] = 1.0
    return t_prob, start_prob


def get_tmat_smat(number_of_pre_states):
    """
    makes the transition matrix and start probabilities for the hmm (initial values)
    """
    #  transition pre states
    x = np.array([])

    for i in range(number_of_pre_states):
        xpre = (1.0 / (number_of_pre_states + 1.0)) * np.ones(number_of_pre_states + 1)
        x = np.concatenate((x, xpre, np.array([0.0])))


    # transition onset states
    xon = np.zeros(number_of_pre_states)
    x = np.concatenate((x, xon, np.array([0.5, 0.5]))) # from a onset state it can stay in onset or go to post

    # transition post states
    xpo = np.zeros(number_of_pre_states + 1)
    x = np.concatenate((x, xpo, np.array([1.0])))

    #transition end state
    #xe = np.zeros(number_of_pre_states + 2)
    #x = np.concatenate((x, xe, np.array([1.0])))
    t_prob = np.reshape(x, (number_of_pre_states + 2, number_of_pre_states + 2))
    start_prob = np.zeros(number_of_pre_states + 2)
    start_prob[0] = 1.0
    return t_prob, start_prob


def get_states(num_pre_states, condition, feature, end=False, start=False, cov_type='diag'):
    pre_mean, pre_cov = get_means_and_vars_of_states('../../lowres_features/', 'pre', condition, feature, cov_type)
    means = np.array([pre_mean])
    covs = np.array([pre_cov])
    while np.shape(means)[0] < num_pre_states:
        m, c = make_copy(pre_mean, pre_cov)
        means = np.vstack((means, [m]))
        covs = np.vstack((covs, [c]))

    onset_mean, onset_cov = get_means_and_vars_of_states('../../lowres_features/', 'onset', condition, feature, cov_type)
    means = np.vstack((means, [onset_mean]))
    covs = np.vstack((covs, [onset_cov]))

    post_mean, post_cov = get_means_and_vars_of_states('../../lowres_features/', 'post', condition, feature, cov_type)
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


if __name__ == "__main__":
    root = '../../lowres_features/'
    train_map = open(root + 'trainset.recs.updated.lowres.cleaned', 'r').readlines()
    train_map = [(line.split('/')[3], line.split('\t')[0], line.split('\t')[1]) for line in train_map]
    n_states = 10
    means, covs = get_states(n_states - 3, 'sirs', 'deviation', end=True)
    tmat, smat = get_tmat_smat_with_end(n_states - 3)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", startprob=smat, transmat=tmat, n_iter=2, init_params='mc')
    for condition, file_path, incident_time in train_map:
        if condition == 'sirs':
        #condition, file_path, incident_time = train_map[110] # a random patient file
            print condition, file_path, incident_time

            t, last_index = overlapped_samples(file_path, incident_reported_time=int(incident_time), overlap=5, window=10, with_end=2)
            if t is None:
                print file_path, 'is bad'
            else:
                model.means_ = means
                model.covars_ = covs
                print 'shape intial', np.shape(covs)
                '''
                best_seq = model.decode(t)
                print 'intial,', best_seq
                print 'final means', model.means_
                print 'initial trans', tmat
                print 'initial startprobs', smat, sum(smat)
                '''
                model.fit([t])
                best_seq = model.decode(t)
                print 'file', file_path
                print 'final,', best_seq
                #print 'final means', model.means_
                #print 'final trans', model.transmat_
                #print 'final startprob', model.startprob_

                if np.isnan(model.means_).any() == False and np.isnan(model.covars_).any() == False:
                    means = model.means_
                    covs = np.array([np.diag(model.covars_[0])])
                    for i in range(1, model.n_components):
                        covs = np.vstack((covs, [np.diag(model.covars_[i])]))
                    print 'shape after', np.shape(covs)
                    tmat = model.transmat_










