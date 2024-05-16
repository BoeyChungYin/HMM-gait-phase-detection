#!/usr/bin/env python3

import numpy as np
import scipy.linalg
from hmm_parameters import trans_mat, start_prob, means, covar

class hmm:

    headers_state = ["time", "torso", "lThigh", "rThigh", "rHeel", "rShank",
                     "torso_vel", "lThigh_vel", "rThigh_vel", "rHeel_vel", "rShank_vel",
                     "torso_acc", "lThigh_acc", "rThigh_acc", "rHeel_acc", "rShank_acc",
                     "lFrontFoot", "lBackFoot", "rFrontFoot", "rBackFoot", "state"]
    
    state_list = {0: "FF",
                  1: "HO",
                  2: "SW",
                  3: "IC"}
    
    # Model Parameters
    trans_mat_log = np.log(np.array(trans_mat))
    start_prob_log = np.log(np.array(start_prob))
    means_vec = np.array(means)
    covar_mat = np.array(covar)

    def __init__(self):

        # self.neutral_pos = [] # only required for threshold alg
        self.curr_obs = [0] * 19
        self.mv_avg = [[0,0,0,0] for _ in range(30)]
        self.sliding_window = []
        self.mv_avg_full = False
        self.sliding_window_full = False
        self.is_start = True
        self.mv_avg_size = 30
        self.window_size = 10
        self.n_states = 4
        self.prev_state = 0
        self.reject_window_size = 10
        self.reject_counter = 0
        self.reject_holding_state = 0
        self.viterbi_matrix = [[0]*self.n_states for i in range(self.window_size)]
        self.back_pointer_matrix = [[0]*self.n_states for i in range(self.window_size)]

    # Stores the new sensor data into the "obs" variable
    def readFromRPi(self, msg):

        self.curr_obs.clear()    # Clear the current observation list

        self.curr_obs.append(msg.torso)
        self.curr_obs.append(msg.hip[0])
        self.curr_obs.append(msg.hip[1])
        self.curr_obs.append(msg.knee[0])
        self.curr_obs.append(msg.knee[1])

        self.curr_obs.append(msg.torso_vel)
        self.curr_obs.append(msg.hip_vel[0])
        self.curr_obs.append(msg.hip_vel[1])
        self.curr_obs.append(msg.knee_vel[0])
        self.curr_obs.append(msg.knee_vel[1])

        self.curr_obs.append(msg.torso_acc)
        self.curr_obs.append(msg.hip_acc[0])
        self.curr_obs.append(msg.hip_acc[1])
        self.curr_obs.append(msg.knee_acc[0])
        self.curr_obs.append(msg.knee_acc[1])

        self.curr_obs.append(msg.foot_sensor[0])
        self.curr_obs.append(msg.foot_sensor[1])
        self.curr_obs.append(msg.foot_sensor[2])
        self.curr_obs.append(msg.foot_sensor[3])
    
        return
    
    # To check if the moving average window is full
    def mvAvgCheck(self):
        if self.mv_avg_full == False:
            # print([0,0]*3)
            if len(self.mv_avg) == self.mv_avg_size:
                self.mv_avg_full = True
        return
    
    # To check if the sliding window is full
    def slidingWindowCheck(self):
        if self.sliding_window_full == False:
            if len(self.sliding_window) == self.window_size:
                self.sliding_window_full = True
        return
    
    # Adds new observations to the moving average window
    def addToMvAvg(self):
        if self.mv_avg_full:
            self.mv_avg.pop(0)
        self.mv_avg.append(self.curr_obs[-4:])
        return

    # Replace the foot sensor data with the moving average foot sensor data
    def mvAvg(self):
        if self.mv_avg_full:
            val = np.cumsum(self.mv_avg, axis=0)[-1] / self.mv_avg_size
            self.curr_obs[-4:] = val
        return

    # Append the current observation to the sliding window
    def slidingWindow(self):
        if self.sliding_window_full:
            self.sliding_window.pop(0)
        self.sliding_window.append(self.curr_obs)
        return
    
    # Applying the viterbi algorithm for the first sequence
    def firstViterbi(self):
        # Initialize the viterbi matrix
        start_em_prob_log = self.multiVarGaussianPDF(self.sliding_window[0])
        for i in range(self.n_states):
            self.viterbi_matrix[0][i] = start_prob[i] + start_em_prob_log[i]

        # Filling up the viterbi matrix and back pointer matrix
        for t in range(1, self.window_size):
            em_prob_log = self.multiVarGaussianPDF(self.sliding_window[t])
            for i in range(self.n_states):
                log_prob, prev_state = max((self.viterbi_matrix[t-1][j] + self.trans_mat_log[j][i] + em_prob_log[i], j) for j in range(self.n_states))
                self.viterbi_matrix[t][i] = log_prob
                self.back_pointer_matrix[t][i] = prev_state

        return
    
    # Applying the viterbi algorithm for subsequent sequences
    def viterbi(self):
        self.viterbi_matrix.pop(0)
        self.back_pointer_matrix.pop(0)

        viterbi_temp = []
        back_pointer_temp = []
        em_prob_log = self.multiVarGaussianPDF(self.curr_obs)
        for i in range(self.n_states):
            log_prob, prev_state = max((self.viterbi_matrix[-1][j] + self.trans_mat_log[j][i] + em_prob_log[i], j) for j in range(self.n_states))
            viterbi_temp.append(log_prob)
            back_pointer_temp.append(prev_state)

        self.viterbi_matrix.append(viterbi_temp)
        self.back_pointer_matrix.append(back_pointer_temp)
        
        return
    
    # Backtrack the result fromt he viterbi algorithm to obtain the besk sequence of states
    def getPrediction(self) -> int:
        # Final step
        best_prob_path, best_last_state = max((self.viterbi_matrix[self.window_size-1][i], i)for i in range(self.n_states))

        # Back tracking to extract best state sequence
        best_path = [best_last_state]   # Intialize the last element of the best sequence
        for t in range(self.window_size-1, 0, -1):
            best_last_state = self.back_pointer_matrix[t][best_last_state]
            best_path.insert(0, best_last_state)
        return best_path[0] # Return the first state in the sliding window that resulted in the best path


    # Obtain a vector with the probabilities of each state given an observation
    def multiVarGaussianPDF(self, obs):
        n = len(self.means_vec) # Dimension of observations (number of different sensor data)
        pdf_vec = []

        for i in range(self.n_states):
            det = np.linalg.det(self.covar_mat[i])
            inv = np.linalg.inv(self.covar_mat[i])

            exp_term = -0.5 * np.dot((obs - self.means_vec[i]).T, np.dot(inv, (obs - self.means_vec[i])))
            norm_term = 1 / (((2*np.pi)**(n/2)) * np.sqrt(det))

            pdf_val = norm_term * np.exp(exp_term)
            pdf_vec.append(pdf_val)

        return np.log(pdf_vec)
    
    # Reject transition until established that it is stable
    def rejectionWindow(self, curr_state: int) -> int:
        legal_transitions = [[1,1,1,0],
                             [1,1,1,0],
                             [1,0,1,1],
                             [1,0,0,1]]

        # Illegal transition
        if legal_transitions[self.prev_state][curr_state] == 0:
            return self.prev_state

        # First occurance of transition
        if curr_state != self.prev_state and self.reject_counter == 0:
            self.reject_holding_state = curr_state
            self.reject_counter += 1
            return self.prev_state
        
        # Another transition occurs during the window (unstable)
        if curr_state != self.reject_holding_state and self.reject_counter > 0:
            self.reject_counter = 0
            return self.prev_state
        
        # Transitions maintains throughout the window (stable)
        if curr_state == self.reject_holding_state and self.reject_counter == self.reject_window_size:
            self.reject_counter = 0
            self.prev_state = curr_state
            return curr_state
        
        # During the transition window
        if curr_state == self.reject_holding_state and self.reject_counter > 0:
            self.reject_counter += 1
            return self.prev_state
        
        # Default when there is no transition
        return self.prev_state

    # Main function that returns the prediction
    def predict(self) -> int:
        self.mvAvgCheck()
        self.slidingWindowCheck()
        self.addToMvAvg()
        self.mvAvg()
        prediction = 0 # Initialize a starting prediction value

        # Once the moving average is in effect, the sliding window for gait prediction starts to run
        if self.mv_avg_full:
            self.slidingWindow()

        if self.sliding_window_full:
            if self.is_start == True:
                self.firstViterbi()
                prediction = self.getPrediction()
                self.is_start = False                
            else:
                self.viterbi()
                prediction = self.getPrediction()

        result = self.rejectionWindow(prediction)

        return result