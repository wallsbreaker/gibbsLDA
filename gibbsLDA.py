import numpy as np
import random

class gibbsLDA(object):
    def __init__(self, doc, voc):
        #document data (list of lists)
        self.documents = doc
        self.vocabulary_size = voc

    def configure(self, alpha, beta, iterations, burn_in, thin_interval):
        self.alpha = alpha
        self.beta = beta

        self.iterations = iterations
        self.burn_in = burn_in
        self.thin_interval = thin_interval

    def initialize(self, topic_num):
        doc_num = self.documents.shape[0]
        self.topic_num = topic_num
        self.word_topic = np.zeros((self.vocabulary_size, topic_num))
        self.document_topic = np.zeros((doc_num, topic_num))
        self.word_topic_sum = np.zeros(topic_num)
        self.word_document_sum = np.zeros(doc_num)

        z_list = []
        for doc  in range(doc_num):
            doc_len = len(self.documents[doc])
            z_doc = np.empty(doc_len)
            for word in range(doc_len):
                topic = random.randint(0, self.topic_num-1)
                z_doc[word] = topic

                word_count = self.documents[doc][word][2]
                self.word_topic[self.documents[doc][word][1]][topic] += word_count
                self.document_topic[doc][topic] += word_count
                self.word_topic_sum[topic] += word_count
                self.word_document_sum[doc] += word_count
            z_list.append(z_doc)
        self.z = np.array(z_list)

    def sample_conditionally(self, doc, doc_word):
        topic = self.z[doc][doc_word]
        word_count = self.documents[doc][doc_word][2]

        self.word_topic[doc_word][topic] -= word_count
        self.document_topic[doc][topic] -= word_count
        self.word_topic_sum[topic] -= word_count
        self.word_document_sum -= word_count

        probability = np.empty(self.topic_num)
        for u in range(self.topic_num):
            probability[u] = (self.word_topic[doc_word][u] + self.beta) / (self.word_topic_sum[u] + self.vocabulary_size *
                                self.beta) * (self.document_topic[doc][u] + self.alpha) / (self.word_document_sum[doc] +
                                    self.topic_num * self.alpha)
        for u in range(1, probability.shape[0]):
            probability[u] += probability[u-1]

        sample = random.random() * probability[self.topic_num-1]
        for u in range(probability.shape[0]):
            if (sample < probability[u]):
                topic = u
                break

        self.word_topic[doc_word][topic] += word_count
        self.document_topic[doc][topic] += word_count
        self.word_topic_sum[topic] += word_count
        self.word_document_sum += word_count

        return topic

    def gibbs(self):
        self.thetasum = np.zeros((self.documents.shape[0], self.topic_num))
        self.phisum = np.zeros((self.topic_num, self.vocabulary_size))
        self.numstats = 0

        for i in range(self.iterations):
            print i
            for j in range(self.z.shape[0]):
                for k in range(self.z[j].shape[0]):
                    topic = self.sample_conditionally(j, k)
                    self.z[j][k] = topic

            if i >= self.burn_in and i % self.thin_interval == 0:
                self.update_params()

    def update_params(self):
        for i in range(self.documents.shape[0]):
            for j in range(self.topic_num):
                self.thetasum[i][j] += (self.document_topic[i][j] + self.alpha) / (self.word_document_sum[i] + self.topic_num * self.alpha)

        for i in range(self.topic_num):
            for j in range(self.vocabulary_size):
                self.phisum[i][j] += (self.word_topic[j][i] + self.beta) / (self.word_topic_sum[i] + self.vocabulary_size * self.beta)

        self.numstats += 1

    def get_theta_and_phi(self):
        if self.numstats == 0:
            return self.thetasum/1.0, self.phisum/1.0
        else:
            return self.thetasum / self.numstats, self.phisum / self.numstats
