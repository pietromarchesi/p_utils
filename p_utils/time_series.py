import numpy as np
import statsmodels.tsa.stattools


def temporal_smoothing(X, y, n):
    # TODO: smooth over boundaries as an option
    if n%2 == 0:
        raise ValueError('The sliding window size n must specify '
                         'an odd number of frames.')
    margin = (n - 1) // 2
    Xn = np.zeros([0, X.shape[1]])
    yn = np.zeros([0])
    for i in range(margin, X.shape[0] - margin):
        ind = range(i-margin,i+margin+1)
        x = np.sum(X[ind,:],axis=0) / float(n)
        Xn = np.vstack((Xn, x))
        yn = np.hstack((yn, y[i]))
    return Xn, yn


def temporal_average(X, y, n):
    Xn = np.zeros([0, X.shape[1]])
    yn = np.zeros([0])
    for i in range(0, X.shape[0] - (n-1),n):
        ind = range(i, i+n)
        if len(set(y[ind])) == 1:
            x = np.sum(X[ind,:],axis=0) / float(n)
            Xn = np.vstack((Xn, x))
            yn = np.hstack((yn, y[i]))
    return Xn, yn


def temporal_expansion(X, y, n):
    if n%2 == 0:
        raise ValueError('The embedding dimension n must specify '
                         'an odd number of frames.')
    margin = (n - 1) // 2
    Xn = np.zeros([0,n*X.shape[1]])
    yn = np.zeros([0])
    for i in range(margin, X.shape[0] - margin):
        ind = range(i-margin,i+margin+1)
        x = np.hstack((X[j,:] for j in ind))
        Xn = np.vstack((Xn,x))
        yn = np.hstack((yn,y[i]))
    return Xn, yn


def temporal_processing(X, y, mode='smooth', n=3):
    options = ['smooth','average','expand','weighted_smooth']
    if mode not in options:
        raise ValueError('The processing mode must be one of the '
                         'following: %s' %options)

    if mode == 'smooth':
        Xp, yp = temporal_smoothing(X, y, n=n)

    if mode == 'average':
        Xp, yp = temporal_average(X, y, n=n)

    if mode == 'expand':
        Xp, yp = temporal_expansion(X, y, n=n)

    return Xp, yp



def autocovariance(X, nlags = 50):
    acv = np.zeros([nlags + 1, X.shape[1]])
    for i in range(X.shape[1]):
        acv[:,i] = statsmodels.tsa.stattools.acovf(X[:,i])[0:nlags + 1]
    return acv


def get_lags(Y):
    lags = np.zeros(Y.shape[1],dtype=int)
    for i in range(Y.shape[1]):
        y = Y[:,i]
        for j in range(y.shape[0] -1):
            if y[j+1] < y[j]:
                lags[i] = j+1
                pass
            if y[j+1] > y[j]:
                lags[i] = j
                break
    return lags


def embed(X, lags):
    mxl  = lags.max()
    emb  = np.zeros_like(X)
    for i in range(mxl, X.shape[0]):
        emb[i, :] = X[np.repeat(i,X.shape[1]) - lags, range(X.shape[1])]

    Xe = np.hstack((X, emb))[mxl:,:]
    return Xe


def temporal_embedding(X, y, nlags = 50):
    acv  = autocovariance(X, nlags = nlags)
    lags = get_lags(acv)
    Xe   = embed(X, lags)
    return Xe, y[y.shape[0] - Xe.shape[0]:]




def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def smooth_array(X, window_len=11, window='hanning'):
    '''
    Smooth every column of the input array X.
    '''
    Xs = []
    for i in range(X.shape[1]):
        Xs.append(smooth(X[:,i], window_len=window_len, window=window))
    return np.array(Xs).transpose()