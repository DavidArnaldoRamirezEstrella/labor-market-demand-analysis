import numpy as np
import sequences.sequence_classifier as sc
import sequences.confusion_matrix as cm
from sequences.log_domain import *
import matplotlib.pyplot as plt
import pdb


class HMMSecondOrder(sc.SequenceClassifier):
    ''' Implements a first order HMM.'''

    def __init__(self, observation_labels, state_labels):
        '''Initialize an HMM. observation_labels and state_labels are the sets
        of observations and states, respectively. They are both LabelDictionary
        objects.'''
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)

        num_states = self.get_num_states()
        num_observations = self.get_num_observations()

        # Vector of probabilities for the initial states: P(state|START,START).
        self.initial_probs = np.zeros([num_states, num_states])

        # Matrix of transition probabilities: P(state|previous_state, previous_previous_state).
        # First index is the state, second index is the previous_state, third index is the previous_previous_state.
        self.transition_probs = np.zeros([num_states, num_states, num_states])

        # Vector of probabilities for the final states: P(STOP|state, previous_state).
        self.final_probs = np.zeros([num_states, num_states])

        # Matrix of emission probabilities. Entry (k,j) is probability
        # of observation k given state j.
        self.emission_probs = np.zeros([num_observations, num_states])

        # Count tables.
        self.initial_counts = np.zeros([num_states, num_states])
        self.transition_counts = np.zeros([num_states, num_states, num_states])
        self.final_counts = np.zeros([num_states, num_states])
        self.emission_counts = np.zeros([num_observations, num_states])


    def train_supervised(self, dataset, smoothing=0):
        ''' Train an HMM from a list of sequences containing observations
        and the gold states. This is just counting and normalizing.'''
        # Set all counts to zeros (optionally, smooth).
        self.clear_counts(smoothing)
        # Count occurrences of events.
        self.collect_counts_from_corpus(dataset)
        # Normalize to get probabilities.
        self.compute_parameters()

    def collect_counts_from_corpus(self, dataset):
        ''' Collects counts from a labeled corpus.'''
        for sequence in dataset.seq_list:
            # Take care of first position.
            # self.initial_counts[sequence.y[0]] += 1
            # self.emission_counts[sequence.x[0], sequence.y[0]] += 1

            # Take care of intermediate positions.
            for i, x in enumerate(sequence.x[2:]):
                y = sequence.y[i+2]
                y_prev = sequence.y[i+1]
                y_prev_prev = sequence.y[i]
                self.emission_counts[x, y] += 1
                self.transition_counts[y, y_prev, y_prev_prev] += 1

            # Take care of last position.
            # self.final_counts[sequence.y[-1]] += 1

    def clear_counts(self, smoothing = 0):
        ''' Clear all the count tables.'''
        self.initial_counts.fill(smoothing)
        self.transition_counts.fill(smoothing)
        self.final_counts.fill(smoothing)
        self.emission_counts.fill(smoothing)

    def compute_parameters(self):
        ''' Estimate the HMM parameters by normalizing the counts.'''
        # Normalize the initial counts.
        # sum_initial = np.sum(self.initial_counts)
        # self.initial_probs = self.initial_counts / sum_initial

        # Normalize the transition counts and the final counts.
        sum_transition = np.sum(self.transition_counts, 0)  # + self.final_counts
        num_states = self.get_num_states()
        self.transition_probs = self.transition_counts / np.tile(sum_transition, [num_states, 1, 1])
        # self.final_probs = self.final_counts / sum_transition

        # Normalize the emission counts.
        sum_emission = np.sum(self.emission_counts, 0)
        num_observations = self.get_num_observations()
        self.emission_probs = self.emission_counts / np.tile(sum_emission, [num_observations, 1])

    def compute_scores_trigram(self, sequence):
        length = len(sequence.x)  # Length of the sequence.
        num_states = self.get_num_states()  # Number of states of the HMM.

        # Initial position.
        initial_scores = [[safe_log(a) for a in b] for b in self.initial_probs]

        # Intermediate position.
        emission_scores = np.zeros([length, num_states]) + logzero()
        transition_scores = np.zeros([length-1, num_states, num_states, num_states]) + logzero()
        for pos in range(length):
            emission_scores[pos, :] = [safe_log(a) for a in self.emission_probs[sequence.x[pos], :]]
            if pos > 0:
                transition_scores[pos-1, :, :, :] = [[[safe_log(a) for a in b] for b in c] for c in self.transition_probs]

        # Final position.
        final_scores = [[safe_log(a) for a in b] for b in self.final_probs]

        return initial_scores, transition_scores, final_scores, emission_scores
        #return transition_scores, emission_scores

    def compute_scores_bigram(self, sequence):
        length = len(sequence.x) # Length of the sequence.
        num_states = self.get_num_states() # Number of states of the HMM.

        # Initial position.
        initial_scores = [safe_log(a) for a in self.initial_probs]

        # Intermediate position.
        emission_scores = np.zeros([length, num_states]) + logzero()
        transition_scores = np.zeros([length-1, num_states, num_states]) + logzero()
        for pos in range(length):
            emission_scores[pos,:] = [safe_log(a) for a in self.emission_probs[sequence.x[pos], :]]
            if pos > 0:
                transition_scores[pos-1,:,:] = [[safe_log(a) for a in b] for b in self.transition_probs]

        # Final position.
        final_scores = [safe_log(a) for a in self.final_probs]

        return initial_scores, transition_scores, final_scores, emission_scores

    ######
    # Plot the transition matrix for a given HMM
    ######
    def print_transition_matrix(self):
        import matplotlib.pyplot as plt
        cax = plt.imshow(self.transition_probs, interpolation='nearest',aspect='auto')
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        plt.xticks(np.arange(0, self.get_num_states()), self.state_labels.names, rotation=90)
        plt.yticks(np.arange(0, self.get_num_states()), self.state_labels.names)
        plt.show()

    def pick_best_smoothing(self,train,test,smooth_values):
        max_smooth = 0
        max_acc = 0
        for i in smooth_values:
               self.train_supervised(train, smoothing=i)
               viterbi_pred_train = self.viterbi_decode_corpus(train)
               posterior_pred_train = self.posterior_decode_corpus(train)
               eval_viterbi_train =   self.evaluate_corpus(train, viterbi_pred_train)
               eval_posterior_train = self.evaluate_corpus(train, posterior_pred_train)
               print ("Smoothing %f --  Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(i,eval_posterior_train,eval_viterbi_train))

               viterbi_pred_test = self.viterbi_decode_corpus(test)
               posterior_pred_test = self.posterior_decode_corpus(test)
               eval_viterbi_test =   self.evaluate_corpus(test, viterbi_pred_test)
               eval_posterior_test = self.evaluate_corpus(test, posterior_pred_test)
               print ("Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(i,eval_posterior_test,eval_viterbi_test))
               if(eval_posterior_test > max_acc):
                   max_acc = eval_posterior_test
                   max_smooth = i
        return max_smooth

    def pick_best_smoothing_bigram(self,train,test,smooth_values):
        max_smooth = 0
        max_acc = 0
        for i in smooth_values:
               self.train_supervised(train, smoothing=i)
               viterbi_pred_train = self.viterbi_decode_corpus_bigram(train)
               eval_viterbi_train = self.evaluate_corpus(train, viterbi_pred_train)
               print ("Smoothing %f --  Train Set Accuracy: Viterbi Decode: %.3f"%(i, eval_viterbi_train))

               viterbi_pred_test = self.viterbi_decode_corpus_bigram(test)
               eval_viterbi_test = self.evaluate_corpus(test, viterbi_pred_test)
               print ("Smoothing %f -- Test Set Accuracy:  Viterbi Decode: %.3f"%(i, eval_viterbi_test))
               if(eval_viterbi_test > max_acc):
                   max_acc = eval_viterbi_test
                   max_smooth = i
               print("")
        return max_smooth

    def graph_error_smoothing(self, train, test, min_smooth, max_smooth, step):
        best_smooth = 0
        max_acc = 0
        X_axis = []  # smooth
        Y_axis = []  # error
        YY_axis = []
        for i in np.arange(min_smooth, max_smooth, step):
               self.train_supervised(train, smoothing=i)
               viterbi_pred_train = self.viterbi_decode_corpus_bigram(train)
               eval_viterbi_train = self.evaluate_corpus(train, viterbi_pred_train)
               #print ("Smoothing %f --  Train Set Accuracy: Viterbi Decode: %.3f"%(i, eval_viterbi_train))

               viterbi_pred_test = self.viterbi_decode_corpus_bigram(test)
               eval_viterbi_test = self.evaluate_corpus(test, viterbi_pred_test)
               print ("Smoothing %f -- Test Set Accuracy:  Viterbi Decode: %.3f"%(i, eval_viterbi_test))
               if(eval_viterbi_test > max_acc):
                   max_acc = eval_viterbi_test
                   best_smooth = i
               print("")
               X_axis.append(i)
               Y_axis.append(100*(1.0-eval_viterbi_test))
               YY_axis.append(100*(1.0-eval_viterbi_train))
        plt.plot(X_axis, Y_axis, 'b', X_axis, YY_axis, 'r')
        plt.show()
        return best_smooth