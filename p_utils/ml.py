import numpy as np
import sklearn.discriminant_analysis

"Machine learning related utilities"


def lda_score(X,y):
    if not set(y) == {0,1}:
        raise ValueError('The label vector y should contain only either 0'
                         'or 1.')
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X, y)
    Xt  = lda.transform(X)
    mt1 = Xt[y == 0, :].mean()
    mt2 = Xt[y == 1, :].mean()
    sc  = np.abs(mt1 - mt2)
    return sc


def lda_score_within_trials(df, contrast, sel = 'all'):

    neurons = [k for k in df.columns if type(k) is int]
    trials = df.trial.unique()

    outcome   = np.zeros(len(trials))
    lda_sc    = np.zeros(len(trials))

    for t in range(len(trials)):
        trial = trials[t]
        outcome[t] = df[df.trial == trial].outcome.unique()[0]
        x = np.array(df[neurons][df.trial == trial])
        if isinstance(sel, list):
            x = x[sel]
        y = np.array(df[df.trial == trial][contrast])

        lda_sc[t] = lda_score(x,y)
        print('Trial %s - %s samples - lda score: %s'
              %(t,x.shape[0],lda_sc[t]))

    return lda_sc, outcome


def lda_debiased(X, y, n_surr):

    lda = lda_score(X, y)

    surr = np.zeros(n_surr)
    for s in range(n_surr):
        ys = np.random.permutation(y)
        surr[s] = lda_score(X, ys)

    return lda, surr.mean()