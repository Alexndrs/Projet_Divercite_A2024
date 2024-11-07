from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        self.NUM_ITERATION_MAX = 3

    def depthFunction(self, step : int):
        '''Compute de depth for the minmax at a given step it is made such that :
           depth = 3 if step < 25
           depth growth exponentially from 25 until last step 40 where depth = 25
        '''
        if step < 30:
            return 3
        
        return int(3 * np.exp((1/(40-25) * np.log(25/3))* (step-25)))
        

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        print("                        _---------------------_                        ")
        print("_______________________/ Inside compute_action \_______________________")

        current_step = current_state.get_step()
        depth = self.depthFunction(current_step)
        print("current_step :", current_step, " - depth :", depth)
        _, action = self.alphaBetaSearch(current_state, depth)
        return action

    def utility(self, s : GameState) -> float:
        return s.scores[self.get_id()] - s.scores[s.next_player.get_id()]

    def isTerminal(self, s : GameState, num_iter : int, num_iter_max : int) -> bool:
        return s.is_done() or num_iter == num_iter_max


    def maxValue(self, s : GameState, alpha : float, beta : float, num_iter : int, num_iter_max : int) -> (float, Action): 
        if self.isTerminal(s, num_iter, num_iter_max):
            return (self.utility(s), None)
        
        score_ = float('-inf')
        action_ = None

        possible_actions = s.generate_possible_heavy_actions()
        for action in possible_actions:
            s_new = action.get_next_game_state()
            score_new, _ = self.minValue(s_new, alpha, beta, num_iter+1, num_iter_max)
            if score_new > score_:
                score_ = score_new
                action_ = action
                alpha = max(alpha, score_)
            if score_ >= beta:
                return score_, action_
        return score_, action_

    def minValue(self, s : GameState, alpha : float, beta : float, num_iter : int, num_iter_max : int) -> (float, Action): 
        if self.isTerminal(s, num_iter, num_iter_max):
            return (self.utility(s), None)
        
        score_ = float('inf')
        action_ = None

        possible_actions = s.generate_possible_heavy_actions()
        for action in possible_actions:
            s_new = action.get_next_game_state()
            score_new, _ = self.maxValue(s_new, alpha, beta, num_iter+1, num_iter_max)
            if score_new < score_:
                score_ = score_new
                action_ = action
                beta = min(beta, score_)
            if score_ <= alpha:
                return score_, action_
        return score_, action_


    def alphaBetaSearch(self, s0 : GameState, num_iter_max : int) -> (float, Action):
        '''
        Effectue un minMax avec pruning à partir de l'état de jeu s0 et avec une profondeur de num_iter_max
        '''
        score, action = self.maxValue(s0, float('-inf'), float('inf'), 0, num_iter_max)
        return score, action