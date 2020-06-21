from test.test import *
from parse import convert_safe

SIZE = 9


def _test_all():
    """
    Runs all tests
    """
    print('Running all tests, make yourself a coffee..\n')

    # tuning
    test_n_prob(SIZE)
    test_beta_prob(SIZE)
    test_itrs_local(SIZE)
    test_itrs_beta(SIZE)

    # separate starts
    test_starts_local(SIZE)
    test_starts_beta(SIZE)

    # performance test
    test_performance_local(SIZE)
    test_performance_beta(SIZE)


def _run_local():
    """
    Runs local hill climbing
    """
    # init board
    b = Board(SIZE)
    b.set_init_state()

    # print puzzle
    b.print()
    print('initial state\n')

    # solve
    solver = HillClimbing(b, max_restarts=100, max_iter=2500, stop_if_found=True)
    sol = solver.try_solve()
    b = sol.board

    # print result
    b.print()
    print(f'Solution found with {sol.result.score} mistakes, number of steps needed: {sol.result.iterations}')


def _run_beta():
    """
    Runs beta hill climbing
    """
    # init board
    b = Board(SIZE)
    b.set_init_state()

    # print puzzle
    b.print()
    print('initial state\n')

    # solve
    solver = CustomBetaHC(b, max_restarts=100, max_iter=2500, n=0.25, beta=0.05, stop_if_found=True)
    sol = solver.try_solve()
    b = sol.board

    # print result
    b.print()
    print(f'Solution found with {sol.result.score} mistakes, number of steps needed: {sol.result.iterations}')


def _get_input() -> int:
    """
    Reads number from 1 to 4
    """
    while True:
        num_str = input()
        num = convert_safe(num_str)
        if num is not None and 1 <= num <= 4:
            return num


def main():
    """
    Program entry point
    """

    while True:
        print('\nWhat do you want to do (enter number from 1 to 4)?\n'
              '1) test\n2) run hill-climbing\n3) run ß-hill-climbing\n4) exit\n')

        num = _get_input()
        if num == 4:
            return
        switch = {
            1: _test_all,
            2: _run_local,
            3: _run_beta
        }
        switch[num]()
        print('Press enter..')
        input()


if __name__ == '__main__':
    main()
