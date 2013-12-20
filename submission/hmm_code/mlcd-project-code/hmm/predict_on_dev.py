__author__ = 'arenduchintala'
import numpy as np
from sklearn.hmm import GaussianHMM
from nearPD import nearPD, nearPSD
import pdb
from pprint import pprint
import matplotlib.pyplot as plt

END = -1e3
START = 1e3
EPS = 1e-10
np.set_printoptions(precision=4, linewidth=120)


def load_full(file_path, num_states, window_size):
    arr = np.loadtxt(file_path)
    reshaped_arr = np.reshape(arr, (num_states, window_size, window_size))
    iter_list = range(num_states)
    iter_list.reverse()
    for i in iter_list:
        if reshaped_arr[i][reshaped_arr[i] > 0.01].shape[0] == 0:
            print 'correcting ith cov:', i, reshaped_arr[i]
            reshaped_arr[i] = 1e-3 * np.identity((reshaped_arr[i].shape[0]))
            #reshaped_arr = np.delete(reshaped_arr, i, 0)
        else:
            try:
                np.linalg.cholesky(reshaped_arr[i])
            except:
                print 'FAILED LOADING FULL COV!'
                print file_path
                print reshaped_arr[i]
                val, vec = np.linalg.eig(reshaped_arr[i])
                val[val < 0] = 1e-3
                dot = np.dot(vec, np.diag(val))
                corrected = np.dot(dot, vec.T)
                print np.linalg.eigvals(corrected)
                np.linalg.cholesky(corrected)
                reshaped_arr[i] = corrected
    print 'completed cholesky check...'
    return reshaped_arr


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


def get_means_and_vars_of_states(root_path, temporal_stage, condition_name, feature):
    """
    root_path path to folder were the stats are located
    temporal_stage folder name of temporal_stage of the incident e.g. pre,onset or post
    condition_name either sevsep, sepshock, or sirs
    feature name : this corresponds to the column of the feats file it can be deviation,skew-(left|right), (slow|fast)-dfa

    """
    constructed_path_mean = root_path + temporal_stage + '_stats/' + condition_name + '-' + feature + '-mean.txt'
    mean = np.loadtxt(constructed_path_mean)
    constructed_path_cov = root_path + temporal_stage + '_stats/' + condition_name + '-' + feature + '-cov.txt'
    cov = np.loadtxt(constructed_path_cov)
    return mean, cov


def make_copy(mean, cov, perturb=True):
    m = np.copy(mean)
    c = np.copy(cov)
    if perturb:
        mp = np.random.multivariate_normal(np.zeros(len(m)), np.identity(len(m)))
        m += mp
    return m, c


def get_untrained_states(num_pre_states, condition, feature, end=False, start=False):
    pre_mean, pre_cov = get_means_and_vars_of_states('../../lowres_features/', 'pre', condition, feature)
    means = np.array([pre_mean])
    covs = np.array([pre_cov])
    while np.shape(means)[0] < num_pre_states:
        m, c = make_copy(pre_mean, pre_cov)
        means = np.vstack((means, [m]))
        covs = np.vstack((covs, [c]))

    onset_mean, onset_cov = get_means_and_vars_of_states('../../lowres_features/', 'onset', condition, feature)
    means = np.vstack((means, [onset_mean]))
    covs = np.vstack((covs, [onset_cov]))

    post_mean, post_cov = get_means_and_vars_of_states('../../lowres_features/', 'post', condition, feature)
    means = np.vstack((means, [post_mean]))
    covs = np.vstack((covs, [post_cov]))

    if end:
        end_mean = END * np.ones(len(pre_mean))
        end_cov = EPS * np.ones(len(pre_mean))
        means = np.vstack((means, [end_mean]))
        covs = np.vstack((covs, [end_cov]))

    if start:
        start_mean = START * np.ones(len(pre_mean))
        start_cov = EPS * np.ones(len(pre_mean))
        means = np.vstack(([start_mean], means))
        covs = np.vstack(([start_cov], covs))
    return means, covs


def get_trained_model(rootpath, condition, n_states, n_iterations, feature, cov_type):
    fname_mean = condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(n_iterations) + '-iter-mean.txt'
    fname_cov = condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(n_iterations) + '-iter-cov.txt'
    fname_tmat = condition + '-cond-' + feature + '-feat-' + str(n_states) + '-states-' + str(n_iterations) + '-iter-transtion.txt'

    constructed_path_mean = rootpath + condition + '/' + fname_mean
    mean = np.loadtxt(constructed_path_mean)
    iter_list = range(n_states)
    iter_list.reverse()
    deleted_means = []
    for i in iter_list:
        if mean[i][mean[i] > 0.01].shape[0] == 0:
            print 'skipping deleting ith mean:', i, mean[i]
            #mean = np.delete(mean, i, 0)
            #deleted_means.append(i)

    constructed_path_cov = rootpath + condition + '/' + fname_cov
    if cov_type == 'full':
        cov = load_full(constructed_path_cov, n_states, 10)
    else:
        cov = np.loadtxt(constructed_path_cov)
    constructed_path_tmat = rootpath + condition + '/' + fname_tmat
    tmat = np.loadtxt(constructed_path_tmat)
    #fixing tmat if any of the means and covs were deleted
    deleted_means.sort()
    deleted_means.reverse()
    for di in deleted_means:
        tmat = np.delete(tmat, di, 1)
        tmat = np.delete(tmat, di, 0)

    smat = np.zeros(tmat.shape[0])
    smat[0] = 1.0
    sum_fix = np.sum(tmat, axis=1)
    sum_fix = 1.0 / sum_fix
    #print tmat
    for i in range(tmat.shape[0]):
        tmat[i] = tmat[i] * sum_fix[i]
        #print 'corrected\n', tmat
    if n_states != tmat.shape[0]:
        print 'removed some states, n_states now corrected to: ', tmat.shape[0], 'was originaly', n_states
        n_states = tmat.shape[0]
    model = GaussianHMM(n_components=n_states, covariance_type=cov_type, startprob=smat, transmat=tmat, n_iter=0, init_params='mc')
    model.means_ = mean
    model.covars_ = cov
    return model


def get_single_patient_overlapped_samples():
    pass


def add_start_and_end_observations(feats, truncate=0):
    if truncate > 0:
        feats = feats[0:truncate]
    return feats


if __name__ == "__main__":
    print 'ok'
    cov_type = 'full'
    root = '../../lowres_features/'
    trained_data_root = '../../lowres_features/trained-' + cov_type + '-cov-no-overlap/'

    dev_map = open(root + 'devset.recs.updated.lowres.cleaned', 'r').readlines()
    dev_map = [(line.split('/')[3], line.split('\t')[0], line.split('\t')[1]) for line in dev_map]

    test_map = open(root + 'testset.recs.updated.lowres.cleaned', 'r').readlines()
    test_map = [(line.split('/')[3], line.split('\t')[0], line.split('\t')[1]) for line in test_map]

    train_map = open(root + 'trainset.recs.updated.lowres.cleaned', 'r').readlines()
    train_map = [(line.split('/')[3], line.split('\t')[0], line.split('\t')[1]) for line in train_map]

    mds = {}
    mds['sirs'] = 20, 100
    mds['sevsep'] = 20, 50
    mds['sepshock'] = 20, 100

    model_index = {}
    model_index['sirs'] = 0
    model_index['sevsep'] = 1
    model_index['sepshock'] = 2
    model_invered = {0: "sirs", 1: "sevsep", 2: "sepshock"}

    sirs_model = get_trained_model(trained_data_root, 'sirs', mds['sirs'][0], mds['sirs'][1], 'deviation', cov_type)
    sevsep_model = get_trained_model(trained_data_root, 'sevsep', mds['sevsep'][0], mds['sevsep'][1], 'deviation', cov_type)
    sepshock_model = get_trained_model(trained_data_root, 'sepshock', mds['sepshock'][0], mds['sepshock'][1], 'deviation', cov_type)

    model_dict = {}
    model_dict['sirs'] = sirs_model
    model_dict['sepshock'] = sepshock_model
    model_dict['sevsep'] = sevsep_model
    model_precision_recall = {"sirs_p": [], "sevsep_p": [], "sepshock_p": [], 'ave_p': [], "sirs_r": [], "sevsep_r": [], "sepshock_r": [],
                              'ave_r': []}

    observation_window = 10
    base = range(10, 1000, 20)
    for observation_window in base:
        print observation_window
        count_mat = np.zeros((3, 3))
        total_counts = np.zeros(3)
        for ex in test_map + dev_map:
            truth = ex[0]
            if truth in model_dict:
                feats = np.loadtxt(ex[1])
                incident_time = int(ex[2].strip())
                t, last_index = overlapped_samples(feats, incident_reported_time=int(incident_time), overlap=0, window=10)
                ready_for_prediction = add_start_and_end_observations(t, observation_window)

                max_ll = float('-inf')
                max_model = ''

                for name, model in model_dict.items():
                    model_ll = model.score(ready_for_prediction)
                    if model_ll > max_ll:
                        max_ll = model_ll
                        max_model = name
                count_mat[model_index[truth]][model_index[max_model]] += 1
        model_results = {}
        sum_c = np.sum(count_mat, 1)
        ave_p = 0.0
        ave_r = 0.0
        for c in range(len(count_mat)):
            col = count_mat[:, c]
            #print col
            row = count_mat[c]
            #print row
            precision = col[c] / np.sum(col)
            recall = col[c] / np.sum(row)
            model_results[c] = (precision, recall)
            ave_p += precision
            ave_r += recall
            model_precision_recall[model_invered[c] + '_p'].append("%.2f" % precision)
            model_precision_recall[model_invered[c] + '_r'].append("%.2f" % recall)
        ave_p = ave_p / 3.0
        ave_r = ave_r / 3.0
        model_precision_recall['ave_p'].append("%.2f" % ave_p)
        model_precision_recall['ave_r'].append("%.2f" % ave_r)
        if observation_window in [270, 290, 310, 330, 350, 370, 390]:
            results = open('results.txt', 'a')
            results.write('\n*** RESULTS ***\n')
            results.write('observation window:' + str(observation_window) + '\n')
            results.write(str(model_index) + '\n')
            results.write(np.array_str(count_mat) + '\n')
            results.write(np.array_str(np.sum(count_mat, 1)) + '\n')
            results.write('model details' + '\n')
            for m_name, m_model in model_dict.items():
                results.write('name:' + m_name + ' iterations:' + str(mds[m_name][1]) + ' states:' + str(m_model.n_components) + '\n')
                msg = '\t precision:' + str(model_results[model_index[m_name]][0]) + '\t recall:' + str(
                    model_results[model_index[m_name]][1])
                results.write(msg + '\n')
            results.write('Ave precision:' + str(ave_p) + ' Ave recall:' + str(ave_r))

pprint(model_precision_recall)

colors = "bgrmyk"
symbols = "o*sd^+"

plt.figure(1)
i = 0
for m, l in model_precision_recall.iteritems():
    if m.startswith('ave'):
        plt.plot(base, l, marker=symbols[i % len(symbols)], c=colors[i % len(colors)], label=str(m))
    i += 1
plt.legend()
plt.xlabel('samples shown to model (min)')
plt.ylabel('precision/recall')
plt.show()

plt.figure(2)
i = 0
for m, l in model_precision_recall.iteritems():
    if not m.startswith('ave'):
        plt.plot(base, l, marker=symbols[i % len(symbols)], c=colors[i % len(colors)], label=str(m))
    i += 1
plt.legend()
plt.xlabel('samples shown to model (min)')
plt.ylabel('precision/recall')
plt.show()