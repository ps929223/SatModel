import numpy as np
import matplotlib.pyplot as plt
def dbscan(data2col, eps, min_samples):
    '''
    출처: 핸즈온머신러닝(2판)
    :param data2col: columns should be more than 2
    :param eps: distance of neighbor
    :param min_samples: minimum sample wihtin eps
    :return: cluster
    eps=0.05
    min_samples=5
    '''
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data2col)
    result={'label':dbscan.labels_,
            'core_sample_idx':dbscan.core_sample_indices_,
            'component':dbscan.components_}
    return result


def knn_class_train(n_neighbors,train_input, train_label):
    '''
    출처: 핸즈온머신러닝(2판)
    :param n_neighbors:
    :param train_input:
    :param train_label:
    :return:
    n_neighbors=1
    train_input=data2col
    train_label=result['label']
    '''
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_input, train_label)
    trained_knn=knn
    return trained_knn

def knn_class_pred_prob(trained_knn, test_input):
    '''
    출처: 핸즈온머신러닝(2판)
    :param trained_knn:
    :param test_input:
    :return:
    test_input=np.array([[-0.5,0],[0,0.5],[1,-0.1],[2,1]])
    '''
    pred=trained_knn.predict(test_input)
    prob=trained_knn.predict_proba(test_input)
    result={'pred':pred,'prob':prob}
    return result

def knn_class_pred_k(trained_knn, test_input, test_label, n_neighbors):
    '''
    출처: 핸즈온머신러닝(2판)
    :param trained_knn:
    :param test_input:
    :param test_label:
    :param n_neighbor:
    :return:
    n_neighbors=1
    test_input=np.array([[-0.5,0],[0,0.5],[1,-0.1],[2,1]])
    test_label=result['label']
    '''
    y_dist, y_pred_idx = trained_knn.kneighbors(test_input, n_neighbors=n_neighbors)
    y_pred = test_label[y_pred_idx]
    y_pred[y_dist > 0.2] = -1
    y_pred.ravel()



'''
Test Code
'''

# from sklearn.datasets import make_moons
# data2col,y=make_moons(n_samples=1000, noise=0.05)

# result=dbscan(data2col=data2col,eps=0.15,min_samples=5)
# plt.scatter(data2col[:,0],data2col[:,1],c=result['label'])