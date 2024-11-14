from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import numpy as np

from seahorse.game.game_layout.board import Piece   #ajoutée manuellement

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
        self.transposition_table = {}

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
        if current_state.in_board((4,4)) :
            print(current_state.get_neighbours(6,6))

        current_step = current_state.get_step()
        depth = self.depthFunction(current_step)
        
        if current_step < 40:
            depth = 20

        if current_step < 30:
            depth = 10

        if current_step < 30:
            depth = 9

        if current_step < 27:
            depth = 5

        if current_step < 20:
            depth = 4

        if current_step < 15:
            depth = 3
        
        if current_step < 2:
            depth = 2

        print("current_step :", current_step, " - depth :", depth)
        _, action = self.alphaBetaSearch(current_state, self.heuristic, depth)
        return action

    def utility(self, s : GameState) -> float:
        return s.scores[self.get_id()] - s.scores[s.next_player.get_id()]

    def isTerminal(self, s : GameState, num_iter : int, num_iter_max : int) -> bool:
        return s.is_done() or num_iter == num_iter_max

    def sortActionsUsingHeuristic(self, possible_actions : [Action], heuristic : callable, playerIsMax : bool):

        sorted_actions = sorted(
            possible_actions,
            key=lambda action: heuristic(action.get_next_game_state()),
            reverse=playerIsMax
        )

        return possible_actions


    def maxValue(self, s : GameState, heuristic : callable, alpha : float, beta : float, num_iter : int, num_iter_max : int) -> (float, Action):
        if self.isTerminal(s, num_iter, num_iter_max):
            return (self.utility(s), None)
        
        score_ = float('-inf')
        action_ = None

        possible_actions = s.generate_possible_heavy_actions()
        # sorted_possible_actions = self.sortActionsUsingHeuristic(list(possible_actions), heuristic, True)

        # for action in sorted_possible_actions:
        for action in possible_actions:
            s_new = action.get_next_game_state()
            score_new, _ = self.minValue(s_new, heuristic, alpha, beta, num_iter+1, num_iter_max)
            if score_new > score_:
                score_ = score_new
                action_ = action
                alpha = max(alpha, score_)
            if score_ >= beta:
                return score_, action_
        return score_, action_

    def minValue(self, s : GameState, heuristic : callable, alpha : float, beta : float, num_iter : int, num_iter_max : int) -> (float, Action): 
        if self.isTerminal(s, num_iter, num_iter_max):
            return (self.utility(s), None)
        
        score_ = float('inf')
        action_ = None

        possible_actions = s.generate_possible_heavy_actions()
        # sorted_possible_actions = self.sortActionsUsingHeuristic(list(possible_actions), heuristic, True)

        # for action in sorted_possible_actions:
        for action in possible_actions:
            s_new = action.get_next_game_state()
            score_new, _ = self.maxValue(s_new, heuristic, alpha, beta, num_iter+1, num_iter_max)
            if score_new < score_:
                score_ = score_new
                action_ = action
                beta = min(beta, score_)
            if score_ <= alpha:
                return score_, action_
        return score_, action_


    def alphaBetaSearch(self, s0 : GameState, heuristic : callable, num_iter_max : int) -> (float, Action):
        '''
        Effectue un minMax avec pruning à partir de l'état de jeu s0 et avec une profondeur de num_iter_max
        '''
        score, action = self.maxValue(s0, heuristic, float('-inf'), float('inf'), 0, num_iter_max)
        return score, action
    
    

    def heuristic(self, s: GameState) -> float:
        delta_score = s.scores[self.get_id()] - s.scores[s.next_player.get_id()]
        step_game = s.get_step()
        pieces_left_player = s.players_pieces_left[self.get_id()]
        pieces_left_opponent = s.players_pieces_left[s.next_player.get_id()]
        
        #ici on veut pénaliser le fait de ne plus avoir certaines couleurs
        num_zero_player = 0
        num_zero_opponent = 0
        for piece in pieces_left_player.keys():
            if pieces_left_player[piece] == 0:
                num_zero_player += 1
            if pieces_left_opponent[piece] == 0:
                num_zero_opponent += 1
        
        #on bonifie les potentielles divercités
        futur_divercity = 0
        for i in range(9):
            for j in range(9):
                color_set = set()
                empty_count = 0
                neighbors = s.get_neighbours(i, j)
                for neighbor in neighbors.values():
                    if isinstance(neighbor[0], Piece):
                        color_set.add(neighbor[0].get_type()[0])
                else:
                    empty_count += 1
                if len(color_set) == 3 and empty_count >= 1 :
                    futur_divercity += 1
                
        
        ponderation = {    # Ponderation des différents critères : peut être différente selon l'avancement du jeu??????
            'delta_score': 3,
            'num_zero': 1,
            'futur_divercity': 5
        }
        
        heuristic_score = ponderation['delta_score'] * delta_score - ponderation['num_zero'] * (num_zero_player - num_zero_opponent) + ponderation['futur_divercity'] * futur_divercity
        
        return(heuristic_score)

