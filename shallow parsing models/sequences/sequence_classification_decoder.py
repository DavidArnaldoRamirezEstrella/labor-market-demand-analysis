import numpy as np
from sequences.log_domain import *
import ipdb, pdb

START = '_START_'
END = '_END_'
START_TAG = '<START>'
END_TAG = '<STOP>'
BR = "**"

NO_LABELS = [
    START,
    END,
    START_TAG,
    END_TAG,
    BR,
]

class SequenceClassificationDecoder():
    ''' Implements a sequence classification decoder.'''

    def __init__(self):
        self.y_dict = {}
        self.x_dict = {}
        self.NO_LABELS_ID = {}

    # #####
    # Computes the forward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    ######
    def init_dicts(self,_x={},_y={}):
        self.x_dict = _x
        self.y_dict = _y
        self.NO_LABELS_ID = [self.y_dict.get_label_id(tag) for tag in NO_LABELS]
        self.NO_LABELS_ID = dict(zip(NO_LABELS,self.NO_LABELS_ID))


    def run_forward(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Forward variables.
        forward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        #forward[0, :] = emission_scores[0, :] + initial_scores
        forward[1,:] = np.zeros(num_states)

        # Forward loop.
        for pos in range(2, length):
            for current_state in range(num_states):
                # Note the fact that multiplication in log domain turns a sum and sum turns a logsum
                forward[pos, current_state] = \
                    logsum(forward[pos - 1, :] + transition_scores[pos - 1, current_state, :])
                forward[pos, current_state] += emission_scores[pos, current_state]

        # Termination.
        #log_likelihood = logsum(forward[length - 1, :] + final_scores)
        log_likelihood = logsum(forward[length - 1, :])

        return log_likelihood, forward


    ######
    # Computes the backward trellis for a given sequence.
    # Receives:
    #
    # Initial scores: (num_states) array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    ######
    def run_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Backward variables.
        backward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        #backward[length - 1, :] = final_scores
        backward[length - 1, :] = np.zeros(num_states)

        # Backward loop.
        for pos in range(length - 2, 0, -1):
            for current_state in range(num_states):
                backward[pos, current_state] = \
                    logsum(backward[pos + 1, :] +
                           transition_scores[pos, :, current_state] +
                           emission_scores[pos + 1, :])

        # Termination.
        #log_likelihood = logsum(backward[1, :] + initial_scores + emission_scores[2, :])
        log_likelihood = logsum(backward[1, :])

        return log_likelihood, backward

    ######
    # Computes the viterbi trellis for a given sequence.
    # Receives:
    #
    # Initial scores: [(num_states),(num_states)] array
    # Transition scores: (length-1, num_states, num_states) array
    # Final scores: (num_states) array
    # Emission scoress: (length, num_states) array
    ######
    def run_viterbi(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(emission_scores, 1) # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states, num_states]) + logzero()
        viterbi_scores[1,:,:] = 0

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Viterbi loop.
        for pos in range(2, length):
            for y in range(num_states):
                for y_1 in range(num_states):
                    viterbi_scores[pos, y, y_1] = \
                        np.max(viterbi_scores[pos - 1, y_1, :] + transition_scores[pos - 2, y, y_1, :])
                    viterbi_scores[pos, y, y_1] += emission_scores[pos, y]
                    viterbi_paths[pos, y, y_1] = \
                        np.argmax(viterbi_scores[pos - 1, y_1, :] + transition_scores[pos - 2, y, y_1, :])
        # Termination.
        y_1 = 0
        y_2 = 0
        mx = 0
        for y in range(num_states):
            for y_1 in range(num_states):
                temp = viterbi_scores[length - 1, y, y_1]
                if (temp > mx):
                    mx = temp
                    y_1 = y
                    y_2 = y_1

        best_score = mx
        best_path[length - 1] = y_1
        best_path[length - 2] = y_2

        # Backtrack.
        for pos in range(length - 3, -1, -1):
            best_path[pos] = viterbi_paths[pos + 2, best_path[pos + 2], best_path[pos + 1]]

        #ipdb.set_trace()
        return best_path, best_score


    def run_viterbi_trigram(self, transition_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(emission_scores, 1)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Initialization.
        for y in range(num_states):
            viterbi_scores[0, y, :] = emission_scores[0, y] + initial_scores[y, :]

        for y in range(num_states):
            for y_1 in range(num_states):
                viterbi_scores[1, y, y_1] = \
                    np.max(viterbi_scores[0, y_1, :] + initial_scores[y, y_1])
                viterbi_scores[1, y, y_1] += emission_scores[1, y]
                viterbi_paths[1, y, y_1] = \
                    np.argmax(viterbi_scores[0, y_1, :] + initial_scores[y, y_1])

        # Viterbi loop.
        for pos in range(2, length):
            for y in range(num_states):
                for y_1 in range(num_states):
                    viterbi_scores[pos, y, y_1] = \
                        np.max(viterbi_scores[pos - 1, y_1, :] + transition_scores[pos - 2, y, y_1, :])
                    viterbi_scores[pos, y, y_1] += emission_scores[pos, y]
                    viterbi_paths[pos, y, y_1] = \
                        np.argmax(viterbi_scores[pos - 1, y_1, :] + transition_scores[pos - 2, y, y_1, :])
        # Termination.
        y_1 = 0
        y_2 = 0
        mx = 0
        for y in range(num_states):
            for y_1 in range(num_states):
                temp = viterbi_scores[length - 1, y, y_1] + final_scores[y, y_1]
                if (temp > mx):
                    mx = temp
                    y_1 = y
                    y_2 = y_1

        best_score = mx
        best_path[length - 1] = y_1
        best_path[length - 2] = y_2

        # Backtrack.
        for pos in range(length - 3, -1, -1):
            best_path[pos] = viterbi_paths[pos + 2, best_path[pos + 2], best_path[pos + 1]]

        #ipdb.set_trace()
        return best_path, best_score


    def run_viterbi_bigram(self, initial_scores, transition_scores, final_scores, emission_scores,sequence):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states]) + logzero()
        #viterbi_scores[1, :] = logzero()
        viterbi_scores[1,self.NO_LABELS_ID[START_TAG]] = 0   # log(1) = 0

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states], dtype=int)
        viterbi_paths[0,:] = self.y_dict.get_label_id(START_TAG)
        viterbi_paths[1,:] = self.y_dict.get_label_id(START_TAG)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Viterbi loop.
        for pos in range(2, length):
            x_name = sequence.sequence_list.x_dict.get_label_name(sequence.x[pos])
            x_1_name = sequence.sequence_list.x_dict.get_label_name(sequence.x[pos-1])

            if x_name==BR or x_name==END:
                fixed_state = 0
                if x_name==BR:
                    fixed_state = self.NO_LABELS_ID[BR]
                else:
                    fixed_state = self.NO_LABELS_ID[END_TAG]

                viterbi_scores[pos, :] = logzero()
                viterbi_scores[pos, fixed_state] = 0

                for current_state in range(num_states):
                    viterbi_paths[pos, current_state] = \
                        np.argmax(viterbi_scores[pos - 1, :] + transition_scores[pos - 1, current_state, :])    
            else:
                for current_state in range(num_states):
                    viterbi_scores[pos, current_state] = \
                        np.max(viterbi_scores[pos - 1, :] + transition_scores[pos - 1, current_state, :])
                    viterbi_scores[pos, current_state] += emission_scores[pos, current_state]
                    if x_1_name == BR:
                        viterbi_paths[pos, current_state] = self.NO_LABELS_ID[BR]
                    else:
                        viterbi_paths[pos, current_state] = \
                            np.argmax(viterbi_scores[pos - 1, :] + transition_scores[pos - 1, current_state, :])
        # Termination.
        best_score = np.max(viterbi_scores[length - 1, :])
        #best_path[length - 1] = np.argmax(viterbi_scores[length - 1, :])
        best_path[length - 1] = self.y_dict.get_label_id(END_TAG)

        # Backtrack.
        for pos in range(length - 2, -1, -1):
            best_path[pos] = viterbi_paths[pos + 1, best_path[pos + 1]]

        return best_path, best_score


    def run_viterbi_bigram_backup(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Initialization.
        viterbi_scores[0, :] = emission_scores[0, :] + initial_scores

        # Viterbi loop.
        for pos in range(1, length):
            for current_state in range(num_states):
                viterbi_scores[pos, current_state] = \
                    np.max(viterbi_scores[pos - 1, :] + transition_scores[pos - 1, current_state, :])
                viterbi_scores[pos, current_state] += emission_scores[pos, current_state]
                viterbi_paths[pos, current_state] = \
                    np.argmax(viterbi_scores[pos - 1, :] + transition_scores[pos - 1, current_state, :])
        # Termination.
        best_score = np.max(viterbi_scores[length - 1, :] + final_scores)
        best_path[length - 1] = np.argmax(viterbi_scores[length - 1, :] + final_scores)

        # Backtrack.
        for pos in range(length - 2, -1, -1):
            best_path[pos] = viterbi_paths[pos + 1, best_path[pos + 1]]

        return best_path, best_score


    def run_forward_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        log_likelihood, forward = self.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
        print('Log-Likelihood =', log_likelihood)

        log_likelihood, backward = self.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
        print('Log-Likelihood =', log_likelihood)

        return forward, backward
