import random
from collections import namedtuple
import numpy as np

from sudoku.board import Board, Pos

INFINITY = float('inf')

"""
Sudoku solution found by a solver
"""
Solution = namedtuple('Solution', 'board result')
"""
Simple structure for one solve-run result
"""
Result = namedtuple('Result', 'score iterations')


class _Solver:
    """
    Represents any sudoku solver
    """

    def __init__(self, board: Board, max_restarts: int, max_iter: int, stop_if_found: bool = False):
        """
        :type board: Sudoku board in initial state
        :type max_restarts: max number of restarts
        :param max_iter: Maximal number of iterations the solver runs
        """
        self._board = board
        self._max_iter = max_iter
        self._max_restarts = max_restarts
        self._stop_if_found = stop_if_found
        # get size of the board
        self._board_size, _ = board.bounds()
        self._results = []

    def try_solve(self):
        """
        Tries to solve the sudoku
        :return: Best solution, iterator to all solutions
        """
        self._board.reset()

        result = self._solve()
        return Solution(self._board, result)

    def all_results(self) -> list:
        """
        :return: List of all results from the previous solve run
        """
        return self._results.copy()

    def _solve(self) -> Result:
        """
        Runs unspecified hill climbing algorithm many times to prevent local minimum problem
        :return Best result from all runs
        """
        self._results = []

        if len(self._board.unfilled_by_row()) == 0:
            # already solved
            return self._solve_trivial()

        best = None
        score_min = INFINITY
        itr_min = INFINITY
        # run hill climbing algorithm repeatedly
        for _ in range(self._max_restarts):
            # run hill climbing
            self._fill()
            score, itr = self._climb()
            # save result from this run
            res = Result(score, itr)
            self._results.append(res)
            # print(f'score: {score}, iterations: {itr}')
            # check if the final solution is found
            if self._stop_if_found and score == 0:
                return res
            # update best run
            if score < score_min:
                score_min = score
                itr_min = itr
                best = self._board.copy()
            if score == score_min and itr < itr_min:
                score_min = score
                itr_min = itr
                best = self._board.copy()
            # restart board for next run
            self._board.reset()

        # no perfect solution found
        self._board = best
        return Result(score_min, itr_min)

    def _solve_trivial(self) -> Result:
        """
        Just fills the board as there is only one way to do it
        :return: Iterator to all solutions
        """
        self._results = [Result(0, 0) for _ in range(self._max_restarts)]
        return Result(0, 0)

    def _climb(self) -> (int, int):
        """
        Hill climbing algorithm
        :return: Score of the best solution
        """
        score_min = self._eval(self._board)
        # keep climbing until max number of iterations is reached
        for i in range(self._max_iter):
            # perform next step on a copy of the board
            board_copy = self._board.copy()
            self._step(board_copy)
            score = self._eval(board_copy)
            # update best result
            if score <= score_min:
                score_min = score
                self._board = board_copy
                # check if the solution was found
                if score == 0:
                    return 0, i + 1
        return score_min, self._max_iter

    def _fill_board_unique(self):
        """
        Fills the board with numbers so that all values are evenly distributed
        Used for solvers which do not modify any values on the board or when there is only one ways to fill the board
        """
        new_board = []
        # fill new board line by line
        for line in self._board.values():
            new_board.append(self._fill_line_unique(line))
        # update the board
        self._board.fill_board(np.array(new_board))

    def _fill_line_unique(self, line: np.array) -> np.array:
        """
        Fills one line with missing numbers so that all numbers are unique
        :param line: Line to be filled
        :return: Filled line
        """
        pool = list(range(1, self._board_size + 1))
        # remove taken numbers from the pool
        for tile in line:
            if tile != 0:
                pool.remove(tile)
        # shuffle the rest
        random.shuffle(pool)
        # fill blanks with numbers from the pool
        return [x if x != 0 else pool.pop() for x in line]

    @staticmethod
    def _swap_in_line(line: np.array, viable):
        """
        Swaps two fields in a line
        :param line: line of numbers
        :param viable: indexes of numbers which can be swapped
        """
        # pick two tiles from the line
        sample = random.sample(viable, 2)
        # swap their position in the original board
        val_1, val_2 = line[sample[0]], line[sample[1]]
        line[sample[0]], line[sample[1]] = val_2, val_1

    @staticmethod
    def _count_unfilled(area):
        """
        :return: Number of filled tiles in the area, list of their indexes
        """
        num = 0
        unfilled = []
        for i, tile in enumerate(area):
            if tile == 0:
                # unfilled tile
                num += 1
                unfilled.append(i)
        return num, unfilled

    @staticmethod
    def _count_mistakes(board: Board):
        """
        :return: Number of mistakes on the board
        """
        mistakes_num, _ = board.check()
        return mistakes_num


class HillClimbing(_Solver):
    """
    Basic hill climbing algorithm
    Introduces exploitation only, uses values swapping within the row to climb
    """

    def _fill(self):
        """
        Fills empty spaces on the board
        """
        self._fill_board_unique()

    def _step(self, board: Board) -> Board:
        """
        Performs one hill climb step
        :param board: Current state of the board to step from
        :return: Board after the step
        """
        # select random line from unfilled ones
        unfilled = board.unfilled_by_row()
        row = list(unfilled.keys())[random.randint(0, len(unfilled) - 1)]
        line = board.values()[row]
        # swap 2 characters in it
        self._swap_in_line(line, unfilled[row])
        # update original board
        board.fill_line(row, line)

        return board

    def _eval(self, board: Board):
        """
        Acts as an heuristic, evaluates current state of the board
        The lower, the better
        :param board: board to be evaluated
        :return: Evaluation score, ideal is 0
        """
        return self._count_mistakes(board)


class _BetaHC(_Solver):
    """
    Beta hill climbing algorithm

    more information about the algorithm:
    https://www.bau.edu.jo/UserPortal/UserProfile/PostsAttach/98216_992_1.pdf
    """

    def __init__(self, board: Board, max_restarts: int, max_iter: int, n, beta, stop_if_found: bool = False):
        """
        :type board: Sudoku board in initial state
        :type max_restarts: max number of restarts
        :param max_iter: Maximal number of iterations the solver runs
        :type n: probability of exploiting a tile
        :param beta: probability of tile mutation
        """
        super(_BetaHC, self).__init__(board, max_restarts, max_iter, stop_if_found)
        self._n_prob = n
        self._beta_prob = beta

    def _generate_fill_number(self) -> int:
        """
        :return: Random number to be placed on a blank spot
        """
        return random.randint(1, self._board_size)

    def _step(self, board: Board) -> Board:
        """
        Performs one ß-hill-climb step
        :param board: Current state of the board to step from
        :return: Board after the step
        """
        # run n-operator and ß-operator on the board
        board = self._neighbouring_operator(board)
        board = self._beta_operator(board)

        return board

    @staticmethod
    def _with_probability(prob: float):
        """
        :return: True with probability of 'prob', otherwise False
        """
        assert 0 <= prob <= 1, 'Probability must be a number from interval <0; 1>'
        return random.random() <= prob


class PaperBetaHC(_BetaHC):
    """
    Modification of hill climbing
    Introduces both exploitation and exploration using two different operators

    more info about the algorithm used for sudoku:
    https://www.researchgate.net/publication/319886025_b-Hill_Climbing_Algorithm_for_Sudoku_Game
    """

    def _fill(self):
        """
        Fills empty spaces on the board
        """
        self._fill_board_random()

    def _fill_board_random(self):
        """
        Fills the board with random values
        Can be used by solvers which do modify values on the board
        """
        # go through all tiles
        for y in range(self._board_size):
            for x in range(self._board_size):
                p = Pos(x=x, y=y)
                # set only blanks
                if not self._board.is_initial(p):
                    new_val = self._generate_fill_number()
                    self._board.set(pos=p, val=new_val)

    def _eval(self, board: Board):
        """
        Acts as an heuristic, evaluates current state of the board
        The lower, the better
        :param board: board to be evaluated
        :return: Evaluation score, ideal is 0
        """
        return self._objective(board)

    def _objective(self, board: Board):
        """
        Objective operator
        Acts as an heuristic, its value represents deviation from the solution)
        The lower the better, ideal value is 0
        :type board: proposed solution
        :return: Heuristic value for the board
        """
        values = board.values()
        dev_sum = 0
        # rows
        for row in values:
            dev_sum += self._eval_tiles(row)
        # columns
        for i in range(self._board_size):
            col = values[:, i]
            dev_sum += self._eval_tiles(col)
        # squares
        for square in board.squares():
            dev_sum += self._eval_tiles(square)

        return dev_sum

    @staticmethod
    def _eval_tiles(tiles: np.array) -> int:
        """
        Helper for the objective function
        :param tiles: tiles to count the summary of
        :return: Deviation
        """
        return abs(45 - np.sum(tiles))

    def _neighbouring_operator(self, board: Board):
        """
        N-operator of the ß-climbing, introduces exploitation to the algorithm
        :param board: State of the board
        :return: Neighbouring state
        """
        values = board.values()

        # iterate modifiable tiles
        for pos in board.unfilled_positions():
            # modify the tile if it meets the probability of n
            if self._with_probability(self._n_prob):
                values[pos.y, pos.x] = self._neighbouring_val(values[pos.y, pos.x])
        # fill the board
        board.fill_board(values)

        return board

    def _neighbouring_val(self, val):
        """
        :return: Neighbouring value (either lower by 1 or higher by 1)
        Handles both over-flow and under-flow
        """
        if self._with_probability(0.5):
            # add 1 (except for UB)
            return val + 1 if val != self._board_size else val
        else:
            # subtract 1 (except for LB)
            return val - 1 if val != 1 else 1

    def _beta_operator(self, board: Board):
        """
        ß-operator of the ß-climbing, introduces exploration to the algorithm
        :param board: Neighbouring state of the board
        :return: New state of the board
        """
        values = board.values()
        # scan the board
        for pos in board.unfilled_positions():
            # regenerate the value if it meets the probability of beta
            if self._with_probability(self._beta_prob):
                values[pos.y, pos.x] = self._generate_fill_number()
        # fill the board
        board.fill_board(values)

        return board


class CustomBetaHC(_BetaHC):
    """
    Modified version of ß-hill-climbing algorithm used for sudoku
    """

    def _fill(self):
        """
        Fills empty spaces on the board
        """
        self._fill_board_unique()

    def _eval(self, board: Board):
        """
        Acts as an heuristic, evaluates current state of the board
        The lower, the better
        :param board: board to be evaluated
        :return: Evaluation score, ideal is 0
        """
        return self._count_mistakes(board)

    def _neighbouring_operator(self, board: Board):
        """
        N-operator of the ß-climbing, introduces exploitation to the algorithm
        :param board: State of the board
        :return: Neighbouring state
        """
        values = board.values()

        # iterate rows
        unfilled = board.unfilled_by_row()
        for row in unfilled:
            # only modify the rows when they meet the probability of n
            if self._with_probability(self._n_prob):
                # swap 2 tiles
                self._swap_in_line(values[row], unfilled[row])
        # update original board
        board.fill_board(values)

        return board

    def _beta_operator(self, board: Board):
        """
        ß-operator of the ß-climbing, introduces exploration to the algorithm
        :param board: Neighbouring state of the board
        :return: New state of the board
        """
        # go through modifiable rows
        unfilled = board.unfilled_by_row()
        for row in unfilled:
            # regenerate the row with probability of beta
            if self._with_probability(self._beta_prob):
                line = board.values()[row]
                # clear
                for index in unfilled[row]:
                    line[index] = 0
                # refill
                board.fill_line(row, np.array(self._fill_line_unique(line)))

        return board




