"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy

class HMM(Classifier):

    def get_model(self): return None
    def set_model(self, model): pass

    model = property(get_model, set_model)

    def __init__(self):
        self.features = 0
        self.labels = 0
        self.feature_list = {}
        self.label2index = {}
        self.index2label = {}

    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function should update self.transtion_count_table
        and self.feature_count_table based on this new given instance

        Add your docstring here explaining how you implement this function

        Returns None
        """
        feature_counter = {}
        for instance in instance_list:
            for feature in instance.features():
                if feature in feature_counter:
                    feature_counter[feature] += 1
                else:
                    feature_counter[feature] = 1
        for instance in instance_list:
            d = []
            for feature in instance.features():
                if feature_counter[feature] < 3:
                    d.append('<UNK>')
                else:
                    d.append(feature)
            instance.data = d


        for instance in instance_list:
            for feature in instance.features():
                if feature in self.feature_list:
                    continue
                else:
                    self.feature_list[feature] = self.features
                    self.features += 1
            instance.feature_vector = self.make_feature_vector(instance)

        self.make_codebooks(instance_list)

        #transition count table
        self.transition_count_table = numpy.ones((self.labels, self.labels))
        self.fill_transition_count_table(instance_list)

        #feature count table
        self.feature_count_table = numpy.ones((self.features, self.labels))
        self.fill_feature_count_table(instance_list)

    def make_codebooks(self, instance_list):
        self.fill_label2index(instance_list)
        self.index2label = {v: k for k, v in self.label2index.iteritems()}

    def fill_label2index(self, instance_list):
        for instance in instance_list:
            for label in instance.label:
                if label in self.label2index:
                    continue
                else:
                    self.label2index[label] = self.labels
                    self.labels += 1

    def fill_transition_count_table(self, instance_list):
        for instance in instance_list:
            for i in range(0, len(instance.label) - 1):
                a = self.label2index[instance.label[i-1]]
                b = self.label2index[instance.label[i]]
                self.transition_count_table[a][b] += 1

    def fill_feature_count_table(self, instance_list):
        for instance in instance_list:
            for i in range(0, len(instance.features()) - 1):
                a = self.feature_list[instance.features()[i]]
                b = self.label2index[instance.label[i]]
                self.feature_count_table[a][b] += 1

    def make_feature_vector(self, instance):
        return [self.feature_list[f] for f in instance.features() if f in self.feature_list]

    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Update codebooks from the given data to be consistent with
        the probability tables

        Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate
        of the appropriate parameters

        Add your docstring here explaining how you implement this function

        Returns None
        """
        self._collect_counts(instance_list)
        self.populate_transition_matrix()
        self.populate_emission_matrix()

    def populate_transition_matrix(self):
        #probability from state a to state b
        self.transition_matrix = self.transition_count_table/self.transition_count_table.sum(axis=1)[:,None]

    def populate_emission_matrix(self):
        self.emission_matrix = self.feature_count_table/self.feature_count_table.sum(axis=0)[None,:]

    def make_feature_vector_classify(self, instance):
        output = []
        for f in instance.features():
            try:
                output.append(self.feature_list[f])
            except KeyError:
                output.append(self.feature_list['<UNK>'])
        return output


    def classify(self, instance):
        """Viterbi decoding algorithm

        Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix

        Add your docstring here explaining how you implement this function

        Returns a list of labels e.g. ['B','I','O','O','B']
        """
        instance.feature_vector = self.make_feature_vector_classify(instance)

        backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        a = len(backtrace_pointers[0][0]) - 1
        f = lambda x: backtrace_pointers[0][x][a]
        l = [x for x in self.index2label]
        y = max(l, key=f)
        pointers = [y]
        for x in reversed(range(1, a)):
            pointers.append(backtrace_pointers[1][y][x])
            y = int(backtrace_pointers[1][y][x])
        best_sequence = list(reversed([self.index2label[x] for x in pointers]))
        print instance.data
        print len(instance.data)
        return best_sequence + ['O']


    # def compute_observation_loglikelihood(self, instance):
    #     """Compute and return log P(X|parameters) = loglikelihood of observations"""
    #     trellis = self.dynamic_programming_on_trellis(instance, True)
    #     loglikelihood = 0.0
    #     return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        Add your docstring here explaining how you implement this function

        Returns trellis filled up with the forward probabilities
        and backtrace pointers for finding the best sequence
        """

        #TODO:Initialize trellis and backtrace pointers
        trellis = numpy.zeros((self.labels, len(instance.feature_vector)))
        backtrace_pointers = numpy.zeros((self.labels, len(instance.feature_vector)))
        #TODO:Traverse through the trellis here
        if run_forward_alg:
            pass
        else:
            f = lambda x: trellis[x][t - 1] * self.transition_matrix[s][x]
            for i in self.index2label:
                trellis[i][0] = self.transition_matrix[0][i] * self.emission_matrix[instance.feature_vector[0]][i]
            for t in range(1, len(instance.feature_vector)):
                for s in self.index2label:
                    trellis[s][t] = max([trellis[k][t-1] * self.transition_matrix[k][s] * self.emission_matrix[instance.feature_vector[t]][s] for k in self.index2label])
                    backtrace_pointers[s][t] = max(self.index2label, key=f)
        print trellis[0]
        print trellis[1]
        return (trellis, backtrace_pointers)

    # def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
    #     """Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)
    #
    #     The algorithm first initializes the model with the labeled data if given.
    #     The model is initialized randomly otherwise. Then it runs
    #     Baum-Welch algorithm to enhance the model with more data.
    #
    #     Add your docstring here explaining how you implement this function
    #
    #     Returns None
    #     """
    #     if labeled_instance_list is not None:
    #         self.train(labeled_instance_list)
    #     else:
    #         #TODO: initialize the model randomly
    #         pass
    #     while True:
    #         #E-Step
    #         self.expected_transition_counts = numpy.zeros((1,1))
    #         self.expected_feature_counts = numpy.zeros((1,1))
    #         for instance in instance_list:
    #             (alpha_table, beta_table) = self._run_forward_backward(instance)
    #             #TODO: update the expected count tables based on alphas and betas
    #             #also combine the expected count with the observed counts from the labeled data
    #         #M-Step
    #         #TODO: reestimate the parameters
    #         if self._has_converged(old_likelihood, likelihood):
    #             break

    # def _has_converged(self, old_likelihood, likelihood):
    #     """Determine whether the parameters have converged or not (EXTRA CREDIT)
    #
    #     Returns True if the parameters have converged.
    #     """
    #     return True

    # def _run_forward_backward(self, instance):
    #     """Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)
    #
    #     Fill up the alpha and beta trellises (the same notation as
    #     presented in the lecture and Martin and Jurafsky)
    #     You can reuse your forward algorithm here
    #
    #     return a tuple of tables consisting of alpha and beta tables
    #     """
    #     alpha_table = numpy.zeros((1,1))
    #     beta_table = numpy.zeros((1,1))
    #     #TODO: implement forward backward algorithm right here
    #
    #     return (alpha_table, beta_table)

