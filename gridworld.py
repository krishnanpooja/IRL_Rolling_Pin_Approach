
import numpy as np
import numpy.random as rn
#from numba import jit
from multiprocessing import Process,Pool
#import itertools
#import multiprocessing
import h5py
from pathlib import Path



#@jit
class Gridworld(object):
    """
    Gridworld MDP.
    """
    #@jit
    def __init__(self, grid_size_x,grid_size_y, wind, discount,border):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((0, 0),(0, 1),(1,1), (-1,1)) #((0, 1), (-1,1), (1,1), (0, 0))
        #((0, 0), (1, 1), (0, 1), (-1, 1)) ## modified actions according to requirement
        self.n_actions = len(self.actions)
        self.n_states = (grid_size_x*grid_size_y)
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.wind = wind
        self.discount = discount
        self.y_scale=0.5
        if grid_size_x==14:
            self.x_scale=1.167
        else:
            self.x_scale=1
        self.border=border       
       
        # Preconstruct the transition probability array.
        self.file='Results/data1_200.h5'
        my_file = Path(self.file)
        self.terminals=[]
        self.transition_probability=np.full((self.n_states,  self.n_actions,self.n_states),0.3,dtype=np.float32)
        self.avg_tp=np.zeros((self.n_states,  self.n_actions,self.n_states),dtype=np.float32)
        self.np_reward=np.full((self.n_states),0.3,dtype=np.float32)
        
        self.file = open("testfile.txt",'a') 

        '''
        if not my_file.exists():
            print('Transition Probabilty file doesnt exists. creating a new file')
           
            self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
           

        else:
            hf = h5py.File(self.file, 'r')
            hf['dataset_1'].read_direct(self.transition_probability)
            hf.close()
        ''' 
    def update_scales(self,x_scale,y_scale):
        self.y_scale=y_scale
        self.x_scale=x_scale
        
    def _trans_prob_irl(self,v_id,trajectories,obs_trajectories):  
        for curr_state, action,next_state in obs_trajectories:
            into_file = '\n'+str(v_id)+','+str(curr_state)+str(action)+str(next_state)
            #print(into_file)
            self.file.write(into_file) 
            self.transition_probability[curr_state,action,next_state]=0
            self.np_reward[curr_state]=0   
                #print('found obstacle',curr_state)
                
        for ego_traj in trajectories:
            for curr_state, action,next_state,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ in ego_traj:
                self.transition_probability[curr_state,action,next_state]=1
                self.np_reward[curr_state]=0.6
                self.np_reward[next_state]=1.0 
                #print('Current State:',curr_state)
                #print('Next State:',next_state)
        self.avg_tp = np.mean([self.transition_probability,self.avg_tp],axis=0)
        
        for state in self.border:
            self.transition_probability[:,:,state] = 0
        for i in self.border:
            self.np_reward[i] = -1
        #print('Transition Probabilty writing into a file')
        #hf = h5py.File(self.file, 'w') 
        #hf.create_dataset('dataset_1', data=self.transition_probability, compression="gzip", compression_opts=9)
        #hf.close()
        
    def _trans_prob(self,ego_trajectories,trajectories):
        for ego_traj in ego_trajectories:
            for curr_state, action,reward,next_state,final_state,_,_,_,_,_,_,_,_ in ego_traj:
                self.transition_probability[curr_state,action,next_state]=self.wind
                self.np_reward[curr_state]=self.wind  
                if next_state==(int(self.point_to_int(final_state))):
                    self.transition_probability[curr_state,action,next_state]=1
                    self.np_reward[next_state]=1
                    print('Fianl state setting reward to 1')
                   
        for obs_traj in trajectories:
            for curr_state, action,_,next_state,_,_,_,_,_,_,_,_,_ in obs_traj:
                self.transition_probability[curr_state,action,next_state]=0
                self.np_reward[curr_state]=0            
                    
        print('Transition Probabilty writing into a file')
        #hf = h5py.File(self.file, 'w') 
        #hf.create_dataset('dataset_1', data=self.transition_probability, compression="gzip", compression_opts=9)
        #hf.close()
        
        #return (ego_trajectories,trajectories)
    '''  
    def get_transition_states_and_probs(self, state, action):
        
        mov_probs = np.zeros([self.n_actions])
        mov_probs[action] = self.wind
        mov_probs += (1-self.wind)/self.n_actions
        for a in range(self.n_actions):
            inc = self.actions[a]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if (nei_s[0] < 0 or nei_s[0] >= self.grid_size) or (nei_s[1] < 0 or nei_s[1] >= self.grid_size):
                # if the move is invalid, accumulates the prob to the current state
                mov_probs[self.n_actions-1] += mov_probs[a]
                mov_probs[a] = 0
                     
        res = []
        for a in range(self.n_actions):
            if mov_probs[a] != 0:
                    inc = self.actions[a]
                    nei_s = (state[0] + inc[0], state[1] + inc[1])
                    res.append((nei_s, mov_probs[a]))
        return res      
        
  
    def get_transition_mat(self):
        #self.transition_probability.fill(0)
        for si in range(0,5,6,11,12,17,18,23,24,29):
            posi = self.int_to_point(si)
            for a in range(self.n_actions):
                probs = self.get_transition_states_and_probs(posi, a)
                for posj, prob in probs:
                    sj = (self.point_to_int(posj))
                    self.transition_probability[si,sj,a] = 0
    '''
    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount) 

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix_irl(self,trajs,objs_traj,feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """
        '''
        features = []
        for n in range(self.n_states):
                f = self.feature_vector(n, "coord")
                features.append(f)
        return np.array(features)
        '''
        features=np.full((self.n_states,18),0.3)
        for traj in trajs:
            for  state,_,_,rel_dist_F,rel_F_vel,rel_dist_F2,rel_F2_vel,rel_dist_B,rel_vel_B,rel_dist_B2,rel_vel_B2,Lane,rel_dist_FL,rel_vel_FL,rel_dist_FR,rel_vel_FR,rel_dist_BR,rel_vel_BR,rel_dist_BL,rel_vel_BL,vel in traj:
                features[state][0]=rel_dist_F
                features[state][1]=rel_F_vel
                features[state][2]=rel_dist_F2
                features[state][3]=rel_F2_vel
                features[state][4]=rel_dist_B
                features[state][5]=rel_vel_B
                features[state][6]=rel_dist_B2
                features[state][7]=rel_vel_B2
                features[state][8]=Lane
                features[state][9]=rel_dist_FL
                features[state][10]=rel_vel_FL
                features[state][11]=rel_dist_FR
                features[state][12]=rel_vel_FR
                features[state][13]=rel_dist_BR
                features[state][14]=rel_vel_BR
                features[state][15]=rel_dist_BL
                features[state][16]=rel_vel_BL
                features[state][17]=vel
        for data in objs_traj:
            features[data[0]][:] = 0
        for state in self.border:
            features[state][:] = -1
        
        return features
    
    def feature_matrix(self,trajs,feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """
        '''
        features = []
        for n in range(self.n_states):
                f = self.feature_vector(n, "coord")
                features.append(f)
        return np.array(features)
        '''
        features=np.zeros((self.n_states,self.grid_size))
        for traj in trajs:
            for  state,_,_,_,_,rel_prec_x,rel_prec_y,rel_prec_vel,rel_foll_x,rel_foll_y,rel_foll_vel,vel,laneID in traj:
                features[state][0]=rel_prec_x
                features[state][1]=rel_prec_y
                features[state][2]=rel_foll_x
                features[state][3]=rel_foll_y
                features[state][4]=rel_prec_vel
                features[state][5]=rel_foll_vel
                features[state][6]=vel
                features[state][7]=laneID
        return features

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """
        x=(i) % (self.x_scale)
        y=(i) / (self.x_scale)
        return ((x),(y))
        #return ((i*17) % self.grid_size), ((i*17) // self.grid_size)
        #return ((i * self.grid_size)//0.46,(i * self.grid_size)//8.77)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """
        #print('p value:',p[0],p[1])
        x = int(p[0]/self.grid_size_x)
        y = int(p[1]/self.grid_size_y)
        #print('computed x and y val:',x,y)
        state = (x+1) + (y*self.x_scale)
        if (state >= self.n_states):
                 print('x,y,p[0],p[1],x1,y1:',x,y,p[0],p[1])
        return state
        #return (p[0]//self.grid_size)*0.46 + (p[1]//self.grid_size)*8.77
        
    def point_to_int_predict(self, p, m2):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """
        x = p[0]
        y = p[1]
        if m2:
           x = x/14
           y = y/20
           x_sc=1.167
        else:
           x = x/9
           y = y/20
           x_sc=1
        state = x + (y*x_sc)
        #if (state >= 40):
        #         print('x,y,p[0],p[1]:',x,y,p[0],p[1])
        return state
        #return (p[0]//self.grid_size)*0.46 + (p[1]//self.grid_size)*8.77

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1   
    
    def _transition_probability(self, i,j,k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """
        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def reward(self, state_int):#final_states
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """
        return self.np_reward[state_int]
        """
        if state_int is int(self.point_to_int(final_states)):
            print('for state state_int:',state_int)
            return 1
        return 0
        """
    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1
  
    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)
