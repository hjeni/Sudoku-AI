"""
Functions for time performance testing
"""

import time
from collections import namedtuple

INFINITY = float('inf')
NAN = float('nan')


def test_time_perfect(solver, runs: int = 1):
    """
    Tests time performance of an algorithm when finding perfect solution
    :param solver: Hill climbing algorithm to be tested
    :param runs: Number of times to find the solution
    :return: best time, worst time, average time, total time
    """
    assert runs >= 1

    time_sum = 0
    time_min = INFINITY
    time_max = -INFINITY
    for _ in range(runs):
        time_curr = 0
        # keep running the algorithm until perfect solution is found
        while True:
            # measure the time
            solution, tmp = _measure_time(solver.try_solve)
            time_curr += tmp
            if solution.result.score == 0:
                break
        # update stats
        time_sum += time_curr
        time_min = time_curr if time_curr < time_min else time_min
        time_max = time_curr if time_curr > time_max else time_max

    # count average time, set to INF if there is no perfect solution
    time_avg = time_sum / runs if time_min != INFINITY else INFINITY
    return time_min, time_max, time_avg, time_sum


def test_time_separate(solver, runs):
    """
    Tests time performance of an algorithm when finding perfect solution
    :param solver: Hill climbing algorithm to be tested
    :param runs: Number of times to restart the algorithm
    :return: best time, worst time, average time, total time
    """
    time_min = INFINITY
    time_max = -INFINITY
    time_sum = 0
    for _ in range(runs):
        # measure time
        _, time_curr = _measure_time(solver.try_solve)
        # update stats
        time_sum += time_curr
        time_min = time_curr if time_curr < time_min else time_min
        time_max = time_curr if time_curr > time_max else time_max

    return time_min, time_max, time_sum / runs, time_sum


def _measure_time(f, *args, **kwargs):
    """
    Measures functions time performance
    :param f: function
    :param args: function args
    :param kwargs: function kwargs
    :return: return value of f, execution time
    """
    start = time.time()
    # run function
    rv = f(*args, **kwargs)
    # count time
    return rv, time.time() - start


"""
Analysis of an algorithm run
accuracy: Ratio of perfect results (between 0 and 1) 
itr_avg: Average number of iterations it took to reach perfect solution
itr_min: Minimal number of iterations it took to reach perfect solution
itr_max: Maximal number of iterations it took to reach perfect solution
itr_list: Sorted list of iterations of all results
"""
RunAnalysis = namedtuple('RunAnalysis', 'accuracy itr_avg itr_min itr_max itr_list')


def analyze_solver(solver) -> RunAnalysis:
    """
    Analyzes separate results of each start of a sudoku solver
    :param solver: Sudoku solver algorithm
    :return: Analysis of the algorithm
    """
    solver.try_solve()
    results = solver.all_results()

    # count analysis stuff
    perfect = [x.iterations for x in results if x.score == 0]
    # sort all perfect results from the run by iterations
    perfect.sort()
    accuracy = len(perfect) / len(results)
    itr_avg = int(sum(perfect) / len(perfect)) if len(perfect) != 0 else NAN
    itr_min = perfect[0] if len(perfect) > 0 else NAN
    itr_max = perfect[-1] if len(perfect) > 0 else NAN

    # compose return value
    return RunAnalysis(accuracy, itr_avg, itr_min, itr_max, perfect)




