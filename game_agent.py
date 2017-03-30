"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return evaluation_function1(game, player)




def evaluation_function1(game, player):
    """ This is identical to improved_score(game, player) in sample_player.py
*************************
 Evaluating: ID_Improved
*************************

Playing Matches:
----------
  Match 1: ID_Improved vs   Random    	Result: 17 to 3
  Match 2: ID_Improved vs   MM_Null   	Result: 15 to 5
  Match 3: ID_Improved vs   MM_Open   	Result: 12 to 8
  Match 4: ID_Improved vs MM_Improved 	Result: 10 to 10
  Match 5: ID_Improved vs   AB_Null   	Result: 15 to 5
  Match 6: ID_Improved vs   AB_Open   	Result: 14 to 6
  Match 7: ID_Improved vs AB_Improved 	Result: 11 to 9


Results:
----------
ID_Improved         67.14%

*************************
   Evaluating: Student
*************************

Playing Matches:
----------
  Match 1:   Student   vs   Random    	Result: 13 to 7
  Match 2:   Student   vs   MM_Null   	Result: 17 to 3
  Match 3:   Student   vs   MM_Open   	Result: 12 to 8
  Match 4:   Student   vs MM_Improved 	Result: 11 to 9
  Match 5:   Student   vs   AB_Null   	Result: 14 to 6
  Match 6:   Student   vs   AB_Open   	Result: 12 to 8
  Match 7:   Student   vs AB_Improved 	Result: 9 to 11


Results:
----------
Student             62.86%


    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))

    return float(own_moves)

def evaluation_function2(game, player):
    """ Chase After the opponent
*************************
 Evaluating: ID_Improved
*************************

Playing Matches:
----------
  Match 1: ID_Improved vs   Random    	Result: 16 to 4
  Match 2: ID_Improved vs   MM_Null   	Result: 15 to 5
  Match 3: ID_Improved vs   MM_Open   	Result: 12 to 8
  Match 4: ID_Improved vs MM_Improved 	Result: 12 to 8
  Match 5: ID_Improved vs   AB_Null   	Result: 14 to 6
  Match 6: ID_Improved vs   AB_Open   	Result: 13 to 7
  Match 7: ID_Improved vs AB_Improved 	Result: 12 to 8


Results:
----------
ID_Improved         67.14%

*************************
   Evaluating: Student
*************************

Playing Matches:
----------
  Match 1:   Student   vs   Random    	Result: 16 to 4
  Match 2:   Student   vs   MM_Null   	Result: 18 to 2
  Match 3:   Student   vs   MM_Open   	Result: 11 to 9
  Match 4:   Student   vs MM_Improved 	Result: 11 to 9
  Match 5:   Student   vs   AB_Null   	Result: 15 to 5
  Match 6:   Student   vs   AB_Open   	Result: 12 to 8
  Match 7:   Student   vs AB_Improved 	Result: 11 to 9


Results:
----------
Student             67.14%

    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - opp_moves)

def evaluation_function3(game, player):
    """ This is identical to improved_score(game, player) in sample_player.py

*************************
 Evaluating: ID_Improved
*************************

Playing Matches:
----------
  Match 1: ID_Improved vs   Random    	Result: 19 to 1
  Match 2: ID_Improved vs   MM_Null   	Result: 12 to 8
  Match 3: ID_Improved vs   MM_Open   	Result: 14 to 6
  Match 4: ID_Improved vs MM_Improved 	Result: 13 to 7
  Match 5: ID_Improved vs   AB_Null   	Result: 12 to 8
  Match 6: ID_Improved vs   AB_Open   	Result: 11 to 9
  Match 7: ID_Improved vs AB_Improved 	Result: 13 to 7


Results:
----------
ID_Improved         67.14%

*************************
   Evaluating: Student
*************************

Playing Matches:
----------
  Match 1:   Student   vs   Random    	Result: 18 to 2
  Match 2:   Student   vs   MM_Null   	Result: 16 to 4
  Match 3:   Student   vs   MM_Open   	Result: 15 to 5
  Match 4:   Student   vs MM_Improved 	Result: 11 to 9
  Match 5:   Student   vs   AB_Null   	Result: 16 to 4
  Match 6:   Student   vs   AB_Open   	Result: 12 to 8
  Match 7:   Student   vs AB_Improved 	Result: 12 to 8


Results:
----------
Student             71.43%

    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # if the depth between moves are different and the scores is same, we'd rather choose the move which has not deeper, because it can finish the game earlier.

    # we can get the depth from the remaining spaces.
    approx_depth = 50 - len(game.get_blank_spaces())
    # i.c) 49 remaining, depth is 1 => 0.01
    #      46 remaining, depth is 4 => 0.04
    #      we'd rather take depth of 1.

    # we don't want to affect the own_moves and opp_moves decision so depth is less than an one.
    return float(own_moves - opp_moves - approx_depth*0.01)

def evaluation_function4(game, player):
    """ This is identical to improved_score(game, player) in sample_player.py
    :param game:
    :param player:
    :return:
*************************
 Evaluating: ID_Improved
*************************

Playing Matches:
----------
  Match 1: ID_Improved vs   Random    	Result: 16 to 4
  Match 2: ID_Improved vs   MM_Null   	Result: 14 to 6
  Match 3: ID_Improved vs   MM_Open   	Result: 11 to 9
  Match 4: ID_Improved vs MM_Improved 	Result: 12 to 8
  Match 5: ID_Improved vs   AB_Null   	Result: 12 to 8
  Match 6: ID_Improved vs   AB_Open   	Result: 9 to 11
  Match 7: ID_Improved vs AB_Improved 	Result: 12 to 8


Results:
----------
ID_Improved         61.43%

*************************
   Evaluating: Student
*************************

Playing Matches:
----------
  Match 1:   Student   vs   Random    	Result: 19 to 1
  Match 2:   Student   vs   MM_Null   	Result: 15 to 5
  Match 3:   Student   vs   MM_Open   	Result: 15 to 5
  Match 4:   Student   vs MM_Improved 	Result: 14 to 6
  Match 5:   Student   vs   AB_Null   	Result: 14 to 6
  Match 6:   Student   vs   AB_Open   	Result: 16 to 4
  Match 7:   Student   vs AB_Improved 	Result: 12 to 8


Results:
----------
Student             75.00%



    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    approx_depth = 49 - len(game.get_blank_spaces())

    # get the position array
    center_spaces = [(3, 3)]

    #if its depth is 3, it's ALWAYS better to move to the center position ( my assumption )
    center_value = 0
    #the game is set to pick random positions for players
    if approx_depth == 3:
        if game.get_player_location(player) in center_spaces:
            center_value = 99999
    return float(center_value + own_moves - opp_moves - approx_depth*0.01)

def evaluation_function5(game, player):
    """ This is identical to improved_score(game, player) in sample_player.py
    :param game:
    :param player:
    :return:
*************************
 Evaluating: ID_Improved
*************************

Playing Matches:
----------
  Match 1: ID_Improved vs   Random    	Result: 15 to 5
  Match 2: ID_Improved vs   MM_Null   	Result: 15 to 5
  Match 3: ID_Improved vs   MM_Open   	Result: 12 to 8
  Match 4: ID_Improved vs MM_Improved 	Result: 10 to 10
  Match 5: ID_Improved vs   AB_Null   	Result: 14 to 6
  Match 6: ID_Improved vs   AB_Open   	Result: 10 to 10
  Match 7: ID_Improved vs AB_Improved 	Result: 11 to 9


Results:
----------
ID_Improved         62.14%

*************************
   Evaluating: Student
*************************

Playing Matches:
----------
  Match 1:   Student   vs   Random    	Result: 14 to 6
  Match 2:   Student   vs   MM_Null   	Result: 16 to 4
  Match 3:   Student   vs   MM_Open   	Result: 14 to 6
  Match 4:   Student   vs MM_Improved 	Result: 14 to 6
  Match 5:   Student   vs   AB_Null   	Result: 16 to 4
  Match 6:   Student   vs   AB_Open   	Result: 13 to 7
  Match 7:   Student   vs AB_Improved 	Result: 13 to 7


Results:
----------
Student             71.43%


    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    approx_depth = 49 - len(game.get_blank_spaces())

    center_spaces = [(3, 3)]

    center_value = 0

    if approx_depth <= 5:
        if game.get_player_location(player) in center_spaces:
            center_value = 99999
    return float(center_value + own_moves - opp_moves - approx_depth*0.01)

def evaluation_function6(game, player):
    """ This is identical to improved_score(game, player) in sample_player.py
    :param game:
    :param player:
    :return:
*************************
 Evaluating: ID_Improved
*************************

Playing Matches:
----------
  Match 1: ID_Improved vs   Random    	Result: 18 to 2
  Match 2: ID_Improved vs   MM_Null   	Result: 17 to 3
  Match 3: ID_Improved vs   MM_Open   	Result: 12 to 8
  Match 4: ID_Improved vs MM_Improved 	Result: 12 to 8
  Match 5: ID_Improved vs   AB_Null   	Result: 12 to 8
  Match 6: ID_Improved vs   AB_Open   	Result: 13 to 7
  Match 7: ID_Improved vs AB_Improved 	Result: 12 to 8


Results:
----------
ID_Improved         68.57%

*************************
   Evaluating: Student
*************************

Playing Matches:
----------
  Match 1:   Student   vs   Random    	Result: 19 to 1
  Match 2:   Student   vs   MM_Null   	Result: 17 to 3
  Match 3:   Student   vs   MM_Open   	Result: 13 to 7
  Match 4:   Student   vs MM_Improved 	Result: 9 to 11
  Match 5:   Student   vs   AB_Null   	Result: 17 to 3
  Match 6:   Student   vs   AB_Open   	Result: 13 to 7
  Match 7:   Student   vs AB_Improved 	Result: 13 to 7


Results:
----------
Student             72.14%


    """


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        best_move = (-1, -1)
        if not legal_moves:
            return (-1, -1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # Iteractive Deepening Search Algorithm
            # https://github.com/aimacode/aima-pseudocode/blob/master/md/Iterative-Deepening-Search.md
            if self.iterative:
                depth = 1
                if self.method == 'minimax':
                    while True:
                        current_max_value, best_move = self.minimax(game, depth)
                        if current_max_value == float("-inf"):
                            break
                        depth += 1
                elif self.method == 'alphabeta':
                    while True:
                        current_max_value, best_move = self.alphabeta(game, depth)
                        if current_max_value == float("-inf"):
                            break
                        depth += 1
                else:
                    print("Not available method")
            else:
                if self.method == 'minimax':
                    _, best_move = self.minimax(game, self.search_depth)
                elif self.method == 'alphabeta':
                    _, best_move = self.alphabeta(game, self.search_depth)
                else:
                    print("Not available method")
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return game.utility(self), (-1, -1)
        # Mini-Max Algorithm :
        # https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        current_max_value = float("-inf")
        current_best_move = (-1, -1)
        current_min_value = float("inf")
        # This is 'Terminal' Test
        if depth == 1 :
            if maximizing_player == True:
                for move in legal_moves:
                    # score of the next move
                    score = self.score(game.forecast_move(move), self)
                    if score > current_max_value:
                        current_max_value, current_best_move = score, move
                return current_max_value, current_best_move
            # Min Node (Beta Node)
            else:
                for move in legal_moves:
                    # score of the next move
                    score = self.score(game.forecast_move(move), self)

                    if score < current_min_value:
                        current_min_value, current_best_move = score, move
                return current_min_value, current_best_move

        if maximizing_player == True:
            for move in legal_moves:
                # score of the next move (Recursive Call)
                # Performs minimax 'depth' times
                score, _ = self.minimax(game.forecast_move(move), depth-1, maximizing_player=False)

                if score > current_max_value:
                    current_max_value, current_best_move = score, move
            return current_max_value, current_best_move
        # Min Node (Beta Node)
        else:
            for move in legal_moves:
                # score of the next move (Recursive Call)
                score, _ = self.minimax(game.forecast_move(move), depth - 1, maximizing_player=True)

                if score < current_min_value:
                    current_min_value, current_best_move = score, move
            return current_min_value, current_best_move

        # Not reachable
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return game.utility(self), (-1, -1)

        # Alpha-beta Algorithm :
        # https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        current_max_value = float("-inf")
        current_best_move = (-1, -1)
        current_min_value = float("inf")
        # This is 'Terminal' Test
        if depth == 1 :
            if maximizing_player == True:
                for move in legal_moves:
                    # score of the next move
                    score = self.score(game.forecast_move(move), self)

                    if score >= beta:
                        return score, move
                    if score > current_max_value:
                        current_max_value, current_best_move = score, move
                return current_max_value, current_best_move
            # Min Node (Beta Node)
            else:
                for move in legal_moves:
                    # score of the next move
                    score = self.score(game.forecast_move(move), self)

                    if score <= alpha:
                        return score, move

                    if score < current_min_value:
                        current_min_value, current_best_move = score, move
                return current_min_value, current_best_move

        if maximizing_player == True:
            for move in legal_moves:
                # score of the next move (Recursive Call)
                # Performs minimax 'depth' times
                score, _ = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, maximizing_player=False)

                if score >= beta:
                    return score, move

                if score > current_max_value:
                    current_max_value, current_best_move = score, move
                alpha = max(alpha, current_max_value)
            return current_max_value, current_best_move
        # Min Node (Beta Node)
        else:
            for move in legal_moves:
                # score of the next move (Recursive Call)
                score, _ = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, maximizing_player=True)

                if score <= alpha:
                    return score, move

                if score < current_min_value:
                    current_min_value, current_best_move = score, move

                beta = min(beta, current_min_value)
            return current_min_value, current_best_move
        raise NotImplementedError
