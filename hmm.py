"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier

class HMM(Classifier):
        
    def get_model(self): return None
    def set_model(self, model): pass

    model = property(get_model, set_model)
		
	def _collect_counts(self, instance_list):
		"""Collect counts necessary for fitting parameters

		This function should update self.transtion_count_table
		and self.feature_count_table based on this new given instance
		
		Add your docstring here explaining how you implement this function

		Returns None
		"""
		pass

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
		self.transition_matrix = numpy.zeros((1,1))
		self.emission_matrix = numpy.zeros((1,1))
		self.transition_count_table = numpy.zeros((1,1))
		self.feature_count_table = numpy.zeros((1,1))
		self._collect_counts(instance_list)
		#TODO: estimate the parameters from the count tables

	def classify(self, instance):
		"""Viterbi decoding algorithm

		Wrapper for running the Viterbi algorithm
		We can then obtain the best sequence of labels from the backtrace pointers matrix

		Add your docstring here explaining how you implement this function

		Returns a list of labels e.g. ['B','I','O','O','B']
		"""
		backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
		best_sequence = []
		return best_sequence

	def compute_observation_loglikelihood(self, instance):
		"""Compute and return log P(X|parameters) = loglikelihood of observations"""
		trellis = self.dynamic_programming_on_trellis(instance, True)
		loglikelihood = 0.0
		return loglikelihood

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
		trellis = numpy.zeros((1,1))
		backtrace_pointers = numpy.zeros((1,1))
		#TODO:Traverse through the trellis here

		return (trellis, backtrace_pointers)

	def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
		"""Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)

		The algorithm first initializes the model with the labeled data if given.
		The model is initialized randomly otherwise. Then it runs 
		Baum-Welch algorithm to enhance the model with more data.

		Add your docstring here explaining how you implement this function

		Returns None
		"""
		if labeled_instance_list is not None:
			self.train(labeled_instance_list)
		else:
			#TODO: initialize the model randomly
			pass
		while True:
			#E-Step
			self.expected_transition_counts = numpy.zeros((1,1))
			self.expected_feature_counts = numpy.zeros((1,1))
			for instance in instance_list:
				(alpha_table, beta_table) = self._run_forward_backward(instance)
				#TODO: update the expected count tables based on alphas and betas
				#also combine the expected count with the observed counts from the labeled data
			#M-Step
			#TODO: reestimate the parameters
			if self._has_converged(old_likelihood, likelihood):
				break

	def _has_converged(self, old_likelihood, likelihood):
		"""Determine whether the parameters have converged or not (EXTRA CREDIT)

		Returns True if the parameters have converged.	
		"""
		return True

	def _run_forward_backward(self, instance):
		"""Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)
	
		Fill up the alpha and beta trellises (the same notation as 
		presented in the lecture and Martin and Jurafsky)
		You can reuse your forward algorithm here

		return a tuple of tables consisting of alpha and beta tables
		"""
		alpha_table = numpy.zeros((1,1))
		beta_table = numpy.zeros((1,1))
		#TODO: implement forward backward algorithm right here

		return (alpha_table, beta_table)

