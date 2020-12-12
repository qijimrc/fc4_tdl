from typing import List, Dict, Tuple
import numpy as np
import copy


class Connect4():
    """ A Connect-4 environment represented by n-tuples for every state.
    """

    def __init__(self,
                n_row:int=6,
                n_col:int=7,
                obs_0: List=None) -> None:
        """ Initialize the board. Set up pieces with
            specific positions if `obs_0` is given else empty.

            Args
              - obs_0: A list of integers from {0, 1, 2}, representing blank, agent1, agent2.
        """
        self.n_row = n_row
        self.n_col = n_col
        self.board = self._init_board(obs_0)



    def step(self, act: Dict) -> Dict:
        """ Move new piece to the board according to one agent's action.
            Return the next observation, reward and indicator denoting
            whether or not the game is done.
        """
        pla = act["player"]
        oc_row, oc_col = act["occupation"]
        # check legal
        if not self._is_legal((oc_row, oc_col)):
            return None
        # update board
        self.board[oc_row][oc_col] = pla
        if oc_row > 0:
            self.board[oc_row-1][oc_col] = 3
        # gen reward
        if self._is_win(act):
            done = True
            reward = 1
        elif self._is_tie():
            done = True
            reward = 0
        else:
            done = False
            reward = 0
        obs = copy.copy(self.board)
        return obs, reward, done


    def reset(self, obs_0: List=None):
        self.board = self._init_board(obs_0)
        board = copy.copy(self.board)
        return board

    def get_obs(self, ):
        board = copy.copy(self.board)
        return board


    def _init_board(self, obs_0: List=None) -> Tuple[List, List]:
        """ Initialize the board. Set up pieces with
            specific positions if `obs_0` is given else empty.

            Return
             - board: A matrix of n_row * n_col including occupations from {0, 1, 2, 3},
                      denoting empty-and-not-reachable, occupated-by-agent1, occupated-by-agent2,
                      and empty-and-reachable.
        """
        n_row, n_col = self.n_row, self.n_col
        board = np.zeros((n_row,n_col), dtype=int)
        if obs_0:
            for k in range(len(obs_0)):
                # board value
                i = self.n_row - 1 - k % self.n_row
                j = k // self.n_row
                board[i][j] = obs_0[k]
        for i in range(n_row):
            for j in range(n_col):
                if board[n_row-i-1][j] == 0:
                    if i==0 or  board[n_row-i][j]==1 or board[n_row-i][j]==2:
                        # change board
                        board[n_row-i-1][j] = 3
        return board


    def _is_legal(self, occ: Tuple[int, int]) -> bool:
        oc_row, oc_col = occ
        if oc_row>=self.n_row or oc_row<0 or oc_col>=self.n_col or oc_col<0:
            return False
        elif self.board[oc_row][oc_col] != 3:
                return False
        return True



    def _is_win(self, act: Dict) -> None or Tuple[int, List]:
        """ Check the board whether or not one of 2 agents has already won.
            Return a tuple denoting winer and positions if there is a winer,
            and None otherwise.
        """
        pla = act["player"]
        oc_row, oc_col = act["occupation"]
        board = self.board

        moves = [((-1,1), (1,-1)), ((0,1), (0,-1)), ((1,1), (-1,-1))]
        for l, r in moves:
            poss = [pla]
            for i in range(3):
                new_row = oc_row + (i+1)*l[0]
                new_col = oc_col + (i+1)*l[1]
                if new_row>=0 and new_row<self.n_row \
                  and new_col>=0 and new_col<self.n_col \
                  and board[new_row][new_col]==pla:
                    poss.append((new_row, new_col))
                else:
                    break
            for i in range(3):
                new_row = oc_row + (i+1)*r[0]
                new_col = oc_col + (i+1)*r[1]
                if new_row>=0 and new_row<self.n_row \
                  and new_col>=0 and new_col<self.n_col \
                  and board[new_row][new_col]==pla:
                    poss.append((new_row, new_col))
                else:
                    break
            if len(poss)>=4:
                return (pla, poss)
        # check straightly down direction
        poss = [pla]
        for i in range(3):
            new_row = oc_row + i + 1
            new_col = oc_col
            if new_row>=0 and new_row<self.n_row \
              and new_col>=0 and new_col<self.n_col \
              and board[new_row][new_col]==pla:
                poss.append((new_row, new_col))
            else:
                break
            if len(poss)>=4:
                return (pla, poss)
        return None


    def _is_tie(self, ):
        for i in range(len(self.board[0])):
            if self.board[0][i]!=1 and self.board[0][i]!=2:
                return False
        return True








