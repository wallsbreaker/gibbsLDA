__author__ = 'tao'
import numpy as np

from itertools import groupby
from nltk.corpus import stopwords

from gibbsLDA import gibbsLDA

def load_data():
    train_file = '/Users/tao/Documents/course/statisticalLearningTheoryAndApplication/homework/3/train.data'
    voc_file = '/Users/tao/Documents/course/statisticalLearningTheoryAndApplication/homework/3/vocabulary.txt'

    train_data = np.loadtxt(train_file, dtype=int, delimiter=' ')
    voc_data = np.loadtxt(voc_file, dtype=str, delimiter=' ')
    return train_data, voc_data

def clean_data(train_data, voc_data):
    return np.array([i for i in train_data if voc_data[i[1]] not in stopwords.words('english')])

def split2document(train_data):
    new_train = []
    for k, g in groupby(train_data, lambda x: x[0]):
        new_train.append(list(g))
    result = np.array(new_train)
    return result

def main():
    train_data, voc_data = load_data()
    train_data = clean_data(train_data, voc_data)
    train_data = split2document(train_data)

    voc_size = voc_data.shape[0]
    lda = gibbsLDA(train_data, voc_size)
    with open('/Users/tao/Documents/course/statisticalLearningTheoryAndApplication/homework/3/result.txt', 'w') as result:
        for k in [5, 10, 20, 30]:
            lda.configure(5.0/k, 0.1, 50, 40, 3)
            lda.initialize(k)
            lda.gibbs()
            theta, phi = lda.get_theta_and_phi()
            phi_result = phi.argmax(axis=1)
            wstr = [str(k)+' ']
            for i in range(k):
                wstr.append(voc_data[phi_result[i]] + ' ')
            wstr.append('\n')
            wstr = ''.join(wstr)
            result.write(wstr)


if __name__ == '__main__':
    main()
