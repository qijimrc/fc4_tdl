from typing import List, Dict, Tuple
import numpy as np
import copy
from collections import defaultdict





class Agent():

    def __init__(self,
                player: int,
                n_state: int=9175000,
                n_vec: int=1120,
                lr: float=0.004,
                gamma: float=0.9,
                e_greed: float=0.1,
                use_mc: bool=False,
                lambda_: float=1,
                REP: bool=True,
                RES: bool=True):
        """
        """
        self.player = player
        self.n_state = n_state
        self.n_vec = n_vec
        self.lr = lr
        self.gamma = gamma
        self.e_greed = e_greed
        self.use_mc = use_mc # whether or not use Monte Carlo method

        # Lookup-Table (n * Q-table)
        # self.LUT = np.random.uniform(0, 1, (n_vec, n_state))
        self._LUT = {}

        # Eligibility Traces
        # self.ET = np.zeros((n_vec, n_state))
        self._ET = {}
        self.lambda_ = lambda_
        self.REP = REP
        self.RES = RES
        self.explored = False # indicate last step

        # Q-table for Q-learning
        self._QT = defaultdict(int)


    def _gen_states(self, obs: List) -> List:
        avail_poss = [] # store left&right
        avail_tuple_state_l = [] # store left
        avail_tuple_state_r = [] # store right
        assert len(obs[0]) % 2 == 1
        mid = len(obs[0]) // 2
        for i in range(len(obs)):
            for j in range(mid, len(obs[0])):
                k = len(obs[0]) - j - 1
                if obs[i][j] != 0:
                    avail_tuple_state_r.append(obs[i][j])
                    if obs[i][j]==3 and j!=k:
                        avail_poss.append(((i,j),("r",len(avail_tuple_state_r)-1)))
                if obs[i][k] != 0:
                    avail_tuple_state_l.append(obs[i][j])
                    if obs[i][k]==3 and j!=k:
                        avail_poss.append(((i,k),("l",len(avail_tuple_state_l)-1)))
                if j==k and obs[i][j]==3:
                        avail_poss.append(((i,k),("r",len(avail_tuple_state_r)-1),
                            ("l",len(avail_tuple_state_l)-1)))
        l_state_idx = 0
        r_state_idx = 0
        for j,v in enumerate(avail_tuple_state_l): l_state_idx += v*np.power(4,j)
        for j,v in enumerate(avail_tuple_state_r): r_state_idx += v*np.power(4,j)
        idx = (l_state_idx, r_state_idx)
        return avail_poss, avail_tuple_state_l, avail_tuple_state_r, idx


    def _gen_next_states(self, obs: List) -> List:
        """ Generate all next available states through n-tuple method.
            Generate two n-tuple for the ** mirrored ** board by the central column.

            Return
              - states: A list of states for current time step, consisting of 3 component:
                         (1) "occupation": the next legal occupation in board.
                         (2) "n_tuple_states": the next state according to occupation.
                         (3) "states-value": the value of next state computed with method of s_t[t_j]*P^j.
        """
        avail_poss, avail_tuple_state_l, avail_tuple_state_r, _ = self._gen_states(obs)
        
        states = []
        l_tuple_state = copy.copy(avail_tuple_state_l)
        r_tuple_state = copy.copy(avail_tuple_state_r)
        for item in avail_poss:
            if len(item) == 2:
                occ, (tp, idx) = item
                if tp=="l":
                    l_tuple_state[idx] = self.player
                else:
                    r_tuple_state[idx] = self.player
            else:
                occ, (rtp,ridx), (ltp,lidx) = item
                assert ltp == "l"
                l_tuple_state[lidx] = self.player
                r_tuple_state[ridx] = self.player
            l_state_idx = 0
            r_state_idx = 0
            for j,v in enumerate(l_tuple_state): l_state_idx += v*np.power(4,j)
            for j,v in enumerate(r_tuple_state): r_state_idx += v*np.power(4,j)
            # states.append((occ, cur_tuple_state, state_idx))
            states.append((occ, (l_state_idx, r_state_idx)))
        return states

    def _get_weights(self, idx: Tuple):
        """ Get weight vector from lookup-table according to tuple-index.
            Return random generated vector and add it to lookup-table if the current `idx` not found.
        """
        mirror_idx = (idx[1], idx[0])
        if (idx not in self._LUT) and (mirror_idx not in self._LUT):
            # self._LUT[idx] = np.random.normal(0, 1, self.n_vec)
            self._LUT[idx] = np.random.uniform(-0.0005, 0.0005, self.n_vec)
            return self._LUT[idx]
        elif idx in self._LUT:
            return self._LUT[idx]
        else:
            return self._LUT[mirror_idx]

    def _set_weights(self, idx: Tuple, value: np.array):
        self._LUT[idx] = value


    def _get_traces(self, idx: int):
        """ Get eligibility traces from ET table.
            Return zero vector and add it to ET table if the current `idx` not found.
        """
        mirror_idx = (idx[1], idx[0])
        if (idx not in self._ET) and (mirror_idx not in self._ET):
            self._ET[idx] = np.zeros(self.n_vec)
            return self._ET[idx]
        elif idx in self._ET:
            return self._ET[idx]
        else:
            return self._ET[idx]

    def _set_traces(self, idx: int, value: np.array):
        self._ET[idx] = value



    def sample(self, obs: List) -> Dict:
        """ Make decision with exploration or exploitation
            according to random probability.
        """
        act = {}
        states = self._gen_next_states(obs)
        occs = [occ for occ, state_idx in states]
        if np.random.uniform(0, 1) < self.e_greed:
            act = self.predict(obs)
            self.explored = False
        else:
            idx = np.random.choice(np.arange(len(occs)))
            act["occupation"] = occs[idx]
            act["player"] = self.player
            self.explored = True
        return act


    def predict(self, obs: List) -> Dict:
        """ Make decision with maximum value function.
        """
        # import ipdb
        # ipdb.set_trace()
        act = {}
        states = self._gen_next_states(obs)
        max_v = -1
        for occ, state_idx in states:
            weights = self._get_weights(state_idx)
            value = np.tanh(weights.sum())
            if value > max_v:
                max_v = value
                max_occ = occ
        # for occ, state_idx in states:
        #     mirror_idx = (state_idx[1], state_idx[0])
        #     if state_idx in self._QT: value = self._QT[state_idx]
        #     elif mirror_idx in self._QT: value = self._QT[mirror_idx]
        # else:
        #     value = 0
        #     if value > max_v:
        #         max_v = value
        #         max_occ = occ
        act["occupation"] = max_occ
        act["player"] = self.player
        act["value"] = max_v
        return act


    def learn_TDLearning(self, obs, act, reward, next_obs, done):
        # get s_t with n-tuple of obs
        _,_,_, state_idx = self._gen_states(obs)
        state_weights = self._get_weights(state_idx)
        state_value = np.tanh(state_weights.sum())

        # get s_{t+1} with n-tuple of next_obs
        _,_,_, next_state_idx = self._gen_states(next_obs)
        next_state_weights = self._get_weights(next_state_idx)
        next_state_value = np.tanh(next_state_weights.sum())
        if done:
            next_state_value = 0

        delta_v = reward + self.gamma*next_state_value - state_value

        if not self.use_mc:
            new_value = state_weights + self.lr*delta_v*(1-np.power(state_value, 2))
            self._set_weights(next_state_idx, new_value)
        else:
            if not self.explored or done:
                new_value = state_weights + self.lr*delta_v*self._get_traces(state_idx)
                self._set_weights(next_state_idx, new_value)

            # update eligibility traces with gradients
            delta_e = 1 - np.power(next_state_value, 2)
            for i in range(len(self._ET)):
                # update
                if i == next_state_idx and self.REP:
                    self._set_traces(i, delta_e)
                else:
                    old_e = self._get_traces(i)
                    self._set_traces(i, self.gamma*self.lambda_*old_e + delta_e)


    def learn_QLearning(self, obs, act, reward, next_obs, done):
        # get s_t with n-tuple of obs
        _,_,_, idx = self._gen_states(obs)
        mirror_idx = (idx[1], idx[0])
        if idx in self._QT:
            state_idx = idx
        elif mirror_idx in self._QT:
            state_idx = mirror_idx
        else:
            state_idx = idx
        state_value = self._QT[state_idx]

        # get s_{t+1} with n-tuple of next_obs
        _,_,_, next_idx = self._gen_states(next_obs)
        mirror_next_idx = (next_idx[1], next_idx[0])
        if next_idx in self._QT:
            next_state_idx = next_idx
        elif mirror_next_idx in self._QT:
            next_state_idx = mirror_next_idx
        else:
            next_state_idx = next_idx
        next_state_value = self._QT[next_state_idx]

        predict_Q = state_value
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * next_state_value
        self._QT[state_idx] += self.lr*(target_Q - predict_Q)

            









