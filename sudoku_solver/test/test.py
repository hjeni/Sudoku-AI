"""
Handles sudoku algorithms time performance testing
"""
from test import _engine
from sudoku.board import Board
from sudoku.solve import HillClimbing, CustomBetaHC, PaperBetaHC

# config
N_PROB_STATIC = 0.25
BETA_PROB_STATIC = 0.05

RESTARTS_LOCAL_STATIC = 50
MAX_ITER_LOCAL_STATIC = 2000

RESTARTS_BETA_STATIC = 5
MAX_ITER_BETA_STATIC = 25000

SOLUTIONS_NUM = 50


def test_performance_local(board_size):
    """
    Tests time performance of local hill climbing using restarts
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting overall time performance of local hill climbing with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # measure times
    solver = HillClimbing(board, 1, MAX_ITER_LOCAL_STATIC, stop_if_found=True)
    times = _engine.test_time_perfect(solver, SOLUTIONS_NUM)

    # print results
    _print_times(*times)


def test_performance_beta(board_size):
    """
    Tests time performance of local hill climbing using restarts
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting overall time performance of beta hill climbing with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # measure times
    solver = CustomBetaHC(board, 1, MAX_ITER_BETA_STATIC, N_PROB_STATIC, BETA_PROB_STATIC, stop_if_found=True)
    times = _engine.test_time_perfect(solver, SOLUTIONS_NUM)

    # print results
    _print_times(*times)


def test_starts_local(board_size):
    """
    Tests time performance of separate local hill climbing (doesn't use restarts)
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting time performance of separate starts of local hill climbing with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # measure times
    solver = HillClimbing(board, 1, MAX_ITER_LOCAL_STATIC)
    times = _engine.test_time_separate(solver, RESTARTS_LOCAL_STATIC)

    # print results
    _print_times(*times)


def test_starts_beta(board_size):
    """
    Tests time performance of separate beta hill climbing runs (doesn't use restarts)
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting time performance of separate starts of beta hill climbing with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # measure times
    solver = CustomBetaHC(board, 1, MAX_ITER_BETA_STATIC, N_PROB_STATIC, BETA_PROB_STATIC)
    times = _engine.test_time_separate(solver, RESTARTS_BETA_STATIC)

    # print results
    _print_times(*times)


def test_itrs_local(board_size):
    """
    Tests different distribution between number of max iterations vs number of restarts in basic hill climbing
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting steps distribution for basic hill climb with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # actual test
    _test_iter(initializer=_create_local, board=board, sum_steps=100000, min_steps=100, growth=2)


def test_itrs_beta(board_size):
    """
    Tests different distribution between number of max iterations vs number of restarts in beta hill climbing
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting steps distribution for basic hill climb with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # actual test
    _test_iter(initializer=_create_beta, board=board, sum_steps=1000000, min_steps=1000, growth=2)


def _test_iter(initializer, board, sum_steps, min_steps, growth):
    """
    Tests iteration distribution for a solver
    Displays the results
    :param initializer: Function which instantiates new solver
    :param board: Test board
    :param sum_steps: summary of steps in all iterations (<number of restarts> * <number of iterations>)
    :param min_steps: minimal number of steps per restart
    :param growth: exponential growth in restarts
    """
    # start with 1 restart, continually increase the number of restarts while distributing the steps among them
    steps = sum_steps
    restarts = 1
    while steps >= min_steps:
        print(f'[restarts = {restarts}, steps = {steps}]:')
        # analyze current steps distribution
        solver = initializer(board, restarts, steps)
        # print the result
        _print_analysis(_engine.analyze_solver(solver))
        # update distribution
        steps //= growth
        restarts *= growth


def _create_local(board, restarts, max_itr):
    """
    :return: Analyzable local hill climbing solver
    """
    return HillClimbing(board, restarts, max_itr)


def _create_beta(board, restarts, max_itr):
    """
    :return: Analyzable beta hill climbing solver
    """
    return CustomBetaHC(board, restarts, max_itr, n=N_PROB_STATIC, beta=BETA_PROB_STATIC)


def test_n_prob(board_size):
    """
    Tests n probability for beta hill climbing
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting steps distribution for basic hill climb with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # start with probability of 40% and decrease
    n = 0.4
    while True:
        print(f'[n probability = {n:.2f}]:')
        # analyze current steps distribution
        solver = CustomBetaHC(board, RESTARTS_BETA_STATIC, MAX_ITER_BETA_STATIC, N_PROB_STATIC, n)
        # print the result
        _print_analysis(_engine.analyze_solver(solver))
        # update beta
        n -= 0.05
        n = round(n, 2)
        if n < 0.01:
            break


def test_beta_prob(board_size):
    """
    Tests beta probability for beta hill climbing
    Displays the results
    :param board_size: size of a board to test on
    """
    # init test board
    board = Board(size=board_size)
    print(f'\nTesting steps distribution for basic hill climb with board of size {board_size}:')
    if not board.set_init_state():
        print('Failed to load the board, quitting..')
        return

    # start with probability of 30% and decrease
    beta = 0.3
    while True:
        print(f'[beta probability = {beta:.2f}]:')
        # analyze current steps distribution
        solver = CustomBetaHC(board, RESTARTS_BETA_STATIC, MAX_ITER_BETA_STATIC, N_PROB_STATIC, beta)
        # print the result
        _print_analysis(_engine.analyze_solver(solver))
        # update beta
        beta = beta - 0.2 if beta > 0.1 else beta - 0.01
        beta = round(beta, 2)
        if beta < 0.01:
            break


def _to_percent(val: float) -> str:
    """
    :return: Percentage representation of a float
    """
    return f'{(val * 100):.4f} %'


def _print_analysis(analysis: _engine.RunAnalysis):
    """
    Prints run analysis
    """
    print(f'\tsteps minimum: {analysis.itr_min}\n'
          f'\tsteps maximum: {analysis.itr_max}\n'
          f'\tsteps average: {analysis.itr_avg}\n'
          f'\taccuracy: {_to_percent(analysis.accuracy)}')


def _print_times(time_min, time_max, time_avg, time_sum):
    """
    Prints time test results
    """
    print(f'\tbest time: {time_min:.4f}s\n'
          f'\tworst time: {time_max:.4f}s\n'
          f'\taverage time: {time_avg:.4f}s\n'
          f'\ttotal time: {time_sum:.2f}s')


