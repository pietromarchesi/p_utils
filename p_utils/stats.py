import numpy as np
import scipy.stats


def paired_t_test_all_comparisons(X, nan_policy = 'propagate'):
    '''
    Compute a paired t test for all comparisons between columns of X.

    Parameters
    ----------

    X : 2D ndarray (n_observations, n_samples)

    Returns
    -------

    res : dict
        Nested dictionary containing the test statistic and pvalue
        for all comparisons between columns of X.

    '''

    N   = X.shape[1]
    res = {i : {k : {} for k in range(N) if k!=i} for i in range(N)}

    for i in range(N):

        for j in res[i].keys():

            test = scipy.stats.ttest_rel(X[:,i], X[:,j],
                                         nan_policy=nan_policy)
            res[i][j]['statistic'] = test.statistic
            res[i][j]['pvalue']    = test.pvalue

            res[j][i]['statistic'] = test.statistic
            res[j][i]['pvalue']    = test.pvalue

    return res

