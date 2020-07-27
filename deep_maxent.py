import numpy as np
import tensorflow as tf
import irl.mdp.gridworld as gridworld
import irl.value_iteration as value_iteration
#import img_utils
import irl.tf_utils as tf_utils

import math
import os
from collections import namedtuple
import time

#os.environ["CUDA_VISIBLE_DEVICE"]="1"

Step = namedtuple('Step','cur_state action reward ')

def normalize(vals):
    """
  normalize to (0, max_val)
  input:
    vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return ((2*(vals - min_val)) / (max_val - min_val))-1


def sigmoid(xs):
    """
  sigmoid function
  inputs:
    xs      1d array
    """
    return [1 / (1 + math.exp(-x)) for x in xs]

class DeepIRLFC:
    def __init__(self, n_input, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc'):
        self.n_input = n_input
        self.lr = lr
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.name = name
        
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
        #self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        self.input_s, self.reward, self.theta = self._build_network(self.name)
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
        self.grad_r = tf.placeholder(tf.float32, [None, 1])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)
        
        #self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
        self.grad_theta = self._compute_gradients(self.reward, self.theta, -self.grad_r)
        # apply l2 loss gradients
        self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

        self.grad_norms = tf.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
        self.sess.run(tf.global_variables_initializer())
        if ((self.name=='F1_F2_FR_B1_BL_BR_m2') or (self.name=='F1_F2_FR_FL_B1_B2_BR_BL_m1')):
           self.saver = tf.train.Saver()
           save_path = self.saver.save(self.sess, "C:/Users/ga67zod/Desktop/Final_IRL"+self.name+".ckpt")
       
    def _compute_gradients(self,tensor, var_list,r):
        grads = tf.gradients(tensor, var_list,r)
        return [grad if grad is not None else tf.zeros_like(var)for var, grad in zip(var_list, grads)]


    def _build_network(self, name):
        input_s = tf.placeholder(tf.float32, [None, self.n_input])
        with tf.variable_scope(name):
            fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.elu,
              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.elu,
              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            reward = tf_utils.fc(fc2, 1, scope="reward")
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, reward, theta


    def get_theta(self):
        return self.sess.run(self.theta)


    def get_rewards(self, states):
        rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
        return rewards


    def apply_grads(self, feat_map, grad_r):
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.n_input])
        _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
        feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
        return grad_theta, l2_loss, grad_norms


def compute_state_visition_freq(P_a, gamma, trajs, policy, obs_trajectories,deterministic=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  
    returns:
    p       Nx1 vector - state visitation frequencies
    """
    N_STATES,N_ACTIONS,_ = np.shape(P_a)
    T = len(trajs[0])
    #print(T)
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.full([N_STATES, T],0)
    #mu = np.full([N_STATES, T],0.3) 
    #for trajectory in obs_trajectories:
            #mu[trajectory[0]] += -1
            
    for traj in trajs: # TODO: add obstacle trajecotry as well
        for step, action,next_state,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ in traj:
            mu[step] += 1
    #for i in [0,5,6,11,12,17,18,23,24,29,30,35,36,41,42,47,48,53,54,59]:
            #mu[i]= -1

    #mu=mu/2

    for s in range(N_STATES):
        for t in range(T-1):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s,int(policy[pre_s]),a] for pre_s in range(N_STATES)])
            else:
                mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s,a1,s]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
    p = np.sum(mu, 1)
    return p


def demo_svf(trajs,obs_trajectories, n_states):
    """
    compute state visitation frequences from demonstrations
  
    input:
    trajs   list of list of Steps - collected from expert
    returns:
    p       Nx1 vector - state visitation frequences   
    """

    p = np.full(n_states,0)
    #print(obs_trajectories)
    #or trajectory in obs_trajectories:
            #p[trajectory[0]] += -1
            
    for traj in trajs: # TODO: add obstacle trajecotry as well
        for step, action,next_state,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ in traj:
            p[step] += 1
    #for i in [0,5,6,11,12,17,18,23,24,29,30,35,36,41,42,47,48,53,54,59]:
            #p[i]= -1
    #p = p/2
    return p

class Deep_Maxent:
    def __init__(self):
            print('')
            
    def deep_maxent_irl(self,ground_r,feat_map, P_a, gamma, trajs,obs_trajectories, lr, n_iters,scope_name,write_true = False):
        """
        Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

        sinputs:
        feat_map    NxD matrix - the features for each state
        P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                           landing at state s1 when taking action 
                                           a at state s0
        gamma       float - RL discount factor
        trajs       a list of demonstrations
        lr          float - learning rate
        n_iters     int - number of optimization steps

        returns
        rewards     Nx1 vector - recoverred state rewards
        """

      # tf.set_random_seed(
        #print(P_a.shape, N_ACTIONS)
      # init nn model
        print('Using GPU:',tf.test.is_gpu_available())
        #nn_r #= DeepIRLFC(feat_map.shape[1], lr, 3, 3)
        self.nn_r = DeepIRLFC(feat_map.shape[1], lr,18 , 18,name=scope_name)
        rewards=self.train(ground_r,feat_map, P_a, gamma, trajs, obs_trajectories,lr, n_iters,write_true)
        return (rewards)

    def train(self,ground_r,feat_map, P_a, gamma, trajs,obs_trajectories, lr, n_iters,write_true):
        # training 
        # find state visitation frequencies using demonstrations
        N_STATES,N_ACTIONS,_  = np.shape(P_a)
        mu_D = demo_svf(trajs,obs_trajectories, N_STATES)
        for iteration in range(n_iters):
            timestamp1 = time.time() 

        # compute the reward matrix
            rewards = self.nn_r.get_rewards(feat_map)
            #print('rewards.shape:',rewards.shape)

        # compute policy 
            #policy = value_iteration.find_policy(N_STATES,N_ACTIONS,P_a, rewards.T, gamma,stochastic=True)
            policy = value_iteration.value_iteration_1(P_a, rewards, gamma)

        # compute expected svf
            mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy,obs_trajectories, deterministic=False)

        # compute gradients on rewards:
            grad_r = ground_r

        # apply gradients to the neural network
            grad_theta, l2_loss, grad_norm = self.nn_r.apply_grads(feat_map, grad_r)
            timestamp2 = time.time()
            #if iteration % (n_iters) == 0:
                #print('iteration:',(iteration))
                #print("Iteration %d took %.2f seconds" % (iteration,timestamp2 - timestamp1))

        rewards = self.nn_r.get_rewards(feat_map)
        return l2_loss,np.matrix(sigmoid((rewards)))
        #return np.matrix((sigmoid(rewards)))





