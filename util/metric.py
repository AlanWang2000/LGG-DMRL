import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import normalized_mutual_info_score, rand_score, adjusted_rand_score
from sklearn.metrics.cluster._supervised import check_clusterings
from sklearn.cluster import KMeans, SpectralClustering
from scipy import sparse as sp
from sklearn.manifold import TSNE


def classify(features, labels, count=10, train_size=0.7, show=True):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_size)
    acc_array = np.zeros(count)
    precision_array = np.zeros(count)
    recall_array = np.zeros(count)
    f1_array = np.zeros(count)
    for i in range(count):
        clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_array[i] = accuracy_score(y_test, y_pred)*100
        precision_array[i], recall_array[i], f1_array[i], _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        precision_array[i], recall_array[i], f1_array[i] = precision_array[i]*100, recall_array[i]*100, f1_array[i] *100
    acc_avg, acc_std = acc_array.mean(), acc_array.std()
    precision_avg, precision_std = precision_array.mean(), precision_array.std()
    f1_avg, f1_std = f1_array.mean(), f1_array.std()

    if show:
        print('Acc= {:.2f} ±{:.2f} % ,  Precision= {:.2f} ±{:.2f} % ,  F1= {:.2f} ±{:.2f} %'
            .format(acc_avg, acc_std, precision_avg, precision_std, f1_avg, f1_std))
    return [acc_avg, acc_std, precision_avg, precision_std, f1_avg, f1_std]

def cluster(n_clusters, features, labels, count=10, method='KMeans', affinity='rbf', show=True):
    pred_all = []
    for i in range(count):
        if method == 'KMeans':
            km = KMeans(n_clusters=n_clusters)
            pred = km.fit_predict(features)
            pred_all.append(pred)
        elif method == 'SC':
            sc = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
            pred = sc.fit_predict(features)
            pred_all.append(pred)
    if show:
        print('{}  '.format(method), end='')
    metrics = get_avg_metric(labels, pred_all, count, show=show)
    return metrics

def get_avg_metric(y_true, y_pred, count=10, show=True):
    nmi_array = np.zeros(count)
    RI_array = np.zeros(count)
    f1_array = np.zeros(count)
    ARI_array = np.zeros(count)
    y_true = y_true.reshape(-1).astype(int)
    if np.min(y_true) == 1:
        y_true -= 1
    for i in range(count):
        y_pred[i] = y_pred[i].reshape(-1).astype(int)
        nmi_array[i] = normalized_mutual_info_score(y_true, y_pred[i])*100
        # RI_array[i] = rand_index_score(y_true, y_pred[i])*100
        RI_array[i] = rand_score(y_true, y_pred[i])*100
        f1_array[i] = b3_precision_recall_fscore(y_true, y_pred[i])*100
        ARI_array[i] = adjusted_rand_score(y_true, y_pred[i])*100
    nmi_avg, nmi_std = nmi_array.mean(), nmi_array.std()
    RI_avg, RI_std = RI_array.mean(), RI_array.std()
    f1_avg, f1_std = f1_array.mean(), f1_array.std()
    ARI_avg, ARI_std = ARI_array.mean(), ARI_array.std()

    if show:
        print('NMI= {:.2f}±{:.2f} % ,  RI= {:.2f}±{:.2f} % ,  f1= {:.2f}±{:.2f} % ,  ARI= {:.2f}±{:.2f} %'
            .format(nmi_avg, nmi_std, RI_avg, RI_std, f1_avg, f1_std, ARI_avg, ARI_std))

    return [nmi_avg, nmi_std, RI_avg, RI_std, f1_avg, f1_std, ARI_avg, ARI_std]


def cluster_acc(Y, Y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size*100


def rand_index_score(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    n_classes = np.unique(labels_true).shape[0]
    n_clusters = np.unique(labels_pred).shape[0]

    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1 or
            n_classes == n_clusters == 0 or
            n_classes == n_clusters == n_samples):
        return 1.0

    # Compute the RI using the contingency data
    contingency = contingency_matrix(labels_true, labels_pred)

    n = np.sum(np.sum(contingency))
    from scipy.special import comb
    t1 = comb(n, 2)
    t2 = np.sum(np.sum(np.power(contingency, 2)))
    nis = np.sum(np.power(np.sum(contingency, 0), 2))
    njs = np.sum(np.power(np.sum(contingency, 1), 2))
    t3 = 0.5 * (nis + njs)

    A = t1 + t2 - t3
    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    AR = (A - nc) / (t1 - nc)
    return A / t1
    



def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.
        .. versionadded:: 0.18
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score


def tsne(Z, Y):
    tsne = TSNE(random_state=0)
    view = tsne.fit_transform(Z)
    n_class = len(np.unique(Y))
    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(view[:, 0], view[:, 1], c=Y.squeeze(), s=200, cmap=plt.cm.get_cmap('Paired', n_class))  # original s=16
    # plt.colorbar(ticks=range(n_class + 1))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
