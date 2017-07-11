import numpy as np
import sklearn.discriminant_analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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


def oob_debiased(X, y, n_estimators=50, n_surr=50):

    rf = RandomForestClassifier(oob_score=True, n_estimators=n_estimators)
    rf.fit(X, y)
    oob = rf.oob_score_

    surr = np.zeros(n_surr)
    for s in range(n_surr):
        ys = np.random.permutation(y)
        rf = RandomForestClassifier(oob_score=True, n_estimators=n_estimators)
        rf.fit(X, ys)
        surr[s] = rf.oob_score_

    return oob, surr.mean()



def cross_val_debiased_score(clf, X, y, cv=5, n_surr=50):

    scores = cross_val_score(clf, X, y, cv=cv)

    surr = np.zeros(n_surr)
    for s in range(n_surr):
        ys = np.random.permutation(y)
        surr_scores = cross_val_score(clf, X, ys, cv=cv)
        surr[s] = surr_scores.mean()

    return scores, surr.mean()

