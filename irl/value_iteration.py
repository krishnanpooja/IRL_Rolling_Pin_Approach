"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import tensorflow as tf
import h5py
from pathlib import Path
import math


def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, k, a] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    #print('computing optimal value')
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = np.matrix(transition_probabilities[s, a, :])
                #print('tp:',tp.shape)
                v_prod = np.matrix(discount*v)
                #print('reward.shape:',((reward.shape)))
                r = np.add(reward,v_prod)
                #print('v_prod.shape:',((v_prod.shape)))
                #print('r.shape:',((r.shape)))
                mul=np.matmul(tp, r.T)
                #print('mul:',mul.shape)
                max_v = max(max_v,mul)

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
                v[s] = max_v
    #print('Done computing optimal value')
    return v
    """
    print('computing optimal value')
    #sess = tf.InteractiveSession()
    
    #init=tf.global_variables_initializer()

        #sess.run(init)
    print('Using GPU:',tf.test.is_gpu_available())
    v = tf.Variable(tf.zeros([n_states]))
    diff = float("inf")
    while diff > threshold:
            diff = 0
            for s in range(n_states):
                    print('Current state:',s)
                    max_v = float("-inf")
                    for a in range(n_actions):
                        tp = tf.convert_to_tensor(transition_probabilities[s, a, :])
                        #print('tp:',(tf.transpose(tp)).shape)
                        #v_prod = tf.scalar_mul(discount,v)
                        #print('reward.shape:',((reward.shape)))
                        r = tf.add(tf.convert_to_tensor(reward),tf.scalar_mul(discount,v))
                        #print('v_prod.shape:',((v_prod.shape)))
                        #print('r.shape:',((r.shape)))
                        mul=tf.tensordot(tp,r,1)
                        init = tf.global_variables_initializer()
                        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                        sess.run(init)
                        res = mul.eval(session=sess)
                        #print('res:',res,s,a)
                        max_v = max(max_v,res)

                    new_diff = abs(sess.run(v[s]) - max_v)
                    if new_diff > diff:
                        diff = new_diff
                        v = tf.scatter_update(v, [s], [max_v])
    print('Done computing optimal value')
    optimal_v = sess.run(v)
    sess.close()
    hf = h5py.File('Results/data1_200.h5', 'w') 
    hf.create_dataset('dataset_1', data=self.transition_probability, compression="gzip", compression_opts=9)
    hf.close()
    return optimal_v
    """

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """
    
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)
    #print(v.shape)
    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = np.matrix(transition_probabilities[i,j,:])
                prod=np.matrix(discount*v)
                #print('prod.shape:',prod.shape)
                val= reward + prod
                #print('val.shape:',val.shape)
                #nval=np.sum(reward,prod)
                #print("After computing reward and discount:",nval.shape)
                Q[i, j] = p.dot(val.T)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        #print('Q.shape',Q.shape)
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy

def value_iteration_1(P_a, rewards, gamma, error=0.01, deterministic=False):
            N_STATES,N_ACTIONS, _  = np.shape(P_a)
            values = np.zeros([N_STATES])
        # estimate values
        #while True:
            values_tmp = values.copy()
            for s in range(N_STATES):
                #print(s)
                for a in range(N_ACTIONS):
                    v_s = []
                    prod = gamma*values_tmp
                    add = rewards + (np.matrix(prod)).T
                    values[s] = np.amax(np.matmul(P_a[s,a,:],add))
                    
                    #values[s] = max([sum([P_a[s, a,s1]*(rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
            e = max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)])
            #print('current error:',e)
            if deterministic:
                # generate deterministic policy
                policy = np.zeros([N_STATES])
                for s in range(N_STATES):
                      policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1]) 
                                      for s1 in range(N_STATES)]) 
                                      for a in range(N_ACTIONS)])
                return policy 
            else:
                # generate stochastic policy
                policy = np.zeros([N_STATES, N_ACTIONS])
                for s in range(N_STATES):
                    for a in range(N_ACTIONS):
                    #v_s = np.array([sum([P_a[s, a,s1]*(rewards[s] + gamma*values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
                        prod = gamma*values
                        #print(rewards.shape,np.matrix(prod).shape)
                        add = rewards + (np.matrix(prod)).T
                        #print(P_a[s,a,:].shape,add.shape)
                        policy[s,a] = np.matmul(P_a[s,a,:],add)
                        #print(policy.shape)
                    policy[s,:] = np.exp(policy[s,:])/np.exp(1+policy[s,:]).sum(axis=0)
                return policy

if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    assert np.isclose(v, opt_v).all()
