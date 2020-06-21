import math
import os
from collections import namedtuple
import numpy as np

from parse import _try_convert_seq

_SAMPLE_INIT_PATH = 'sample_data/sample_init_'
_SAMPLE_INIT_FORMAT = '.txt'


"""
Position on the bard
"""
Pos = namedtuple('Pos', 'x y')


class Board:
    """
    Customizable sudoku table
    """

    def __init__(self, size: int = 9, display_blank: bool = False):
        assert int(math.sqrt(size)) ** 2 == size, 'size of the board must be a square'

        self._display_blank = display_blank
        # create empty board
        self._size = size
        self._size_sq = int(math.sqrt(size))
        self._board = np.full(shape=(size, size), fill_value=0)
        # save initial state
        self._board_init = None
        # count digits to print
        self._max_digits = self._count_digits(size)

        # dictionary of indexes of lines which contain at least 2 blanks
        # format: <unfilled line index> : <list of indexes of unfilled tiles>
        self._unfilled = {}

    def set_init_state(self):
        """
        Initializes the board with initial state
        :return: Success
        """
        path = self._assemble_path()
        if not os.path.exists(path):
            return False
        # load from file
        if self._init_from_file(path):
            # save initial state
            self._board_init = self._board.copy()
            # init unfilled
            for i, line in enumerate(self._board_init):
                num, indexes = self._count_unfilled(line)
                if num >= 2:
                    self._unfilled[i] = indexes
            return True

        return False

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

    def unfilled_by_row(self) -> dict:
        """
        :return: Dictionary with row numbers as keys and unfilled indexes as values
        """
        return self._unfilled.copy()

    def unfilled_positions(self) -> list:
        """
        :return: List of all unfilled tiles
        """
        positions = []
        # iterate row numbers
        for row in self._unfilled:
            # iterate indexes of the row
            for index in self._unfilled[row]:
                # add to the list
                positions.append(Pos(x=index, y=row))
        return positions

    def values(self) -> np.array:
        """
        :return: 2D array of all values
        """
        return self._board.copy()

    def at(self, pos: Pos):
        """
        :return: Value on given position, None when the position is out of bounds
        """
        if not self.is_valid(pos):
            return None
        return self._board[pos.y, pos.x]

    def is_valid(self, pos: Pos):
        """
        :return: True when the positions is within the board, otherwise False
        """
        return 0 <= pos.x < self._size and 0 <= pos.y < self._size

    def is_initial(self, pos: Pos):
        """
        :return: True when the value on given position was set initially, otherwise False
        """
        if not self.is_valid(pos):
            return False
        return self._board_init[pos.y, pos.x] != 0

    def fill_board(self, values: np.array):
        """
        Tries to reset all values on the board with passed ones
        :return: Success
        """
        # check bounds
        if values.shape != (self._size, self._size):
            return False
        # scan all tiles first
        for line_init, line_new in zip(self._board_init, values):
            for tile_init, tile_new in zip(line_init, line_new):
                # check if any initial value was changed
                if tile_init != 0 and tile_init != tile_new:
                    return False
        # initial values match
        self._board = values
        return True

    def fill_line(self, line_num: int, values: np.array):
        """
        Tries to reset all values within one line with passed ones
        :return: Success
        """
        # check bounds
        if len(values) != self._size or not (0 <= line_num < self._size):
            return False
        # scan the line
        for tile_init, tile_new in zip(self._board_init[line_num], values):
            # check if any initial value was changed
            if tile_init != 0 and tile_init != tile_new:
                return False
        # initial values match
        self._board[line_num] = values

    def set(self, pos: Pos, val: int):
        """
        Tries to set given tile to given value
        :return: Success
        """
        if not self.is_valid(pos) or self.is_initial(pos):
            return False
        self._board[pos.y, pos.x] = val
        return True

    def reset(self):
        """
        Resets the board to initial state
        """
        self._board = self._board_init.copy()

    def bounds(self) -> (int, int):
        """
        :return: size of the board, size of a square area
        """
        return self._size, self._size_sq

    def print(self, offset: int = 0):
        """
        Prints the board
        :param offset: Number of lines before the board is printed
        """
        print('\n' * offset)

        # initial border
        self._print_dlm_horizontal()
        # line by line
        for i, line in enumerate(self._board):
            # line
            self._print_line(line)
            # delimiter
            if not i == 0 and i % self._size_sq == self._size_sq - 1:
                self._print_dlm_horizontal()
        # edge case
        if self._size == 1:
            self._print_dlm_horizontal()

    def copy(self):
        """
        :return: Shallow copy of the board
        """
        b = Board(self._size, self._display_blank)
        b._board = self._board.copy()
        b._board_init = self._board_init.copy()
        b._unfilled = self._unfilled.copy()
        return b

    def _print_line(self, line: np.array):
        """
        Prints one line of the board
        """
        boarder = '| '
        to_print = boarder
        for i, x in enumerate(line):
            # set delimiter
            dlm = boarder if i % self._size_sq == self._size_sq - 1 else ''
            # set printable character
            val = self._offset_number(x) if x != 0 or self._display_blank else ' '
            to_print += f'{val} {dlm}'
        print(to_print)

    def _print_dlm_horizontal(self, dlm: str = '-'):
        """
        Prints horizontal delimiting line
        """
        # digits + vertical delimiters + 1
        num = int(self._size * (1 + self._max_digits)) + int(self._size_sq * 2) + 1
        print(dlm * num)

    def _init_from_file(self, path):
        """
        Initializes the board with data from a file
        When the initialization fails, anything can be stored in the board
        :param path: file path
        :return: Success
        """
        line_num = 0
        with open(path, 'r') as f:
            for line in f:
                # convert line to sudoku line
                data = _try_convert_seq(line)
                # error check
                if None in data or len(data) != self._size:
                    return False
                # set the line
                self._board[line_num] = np.array(data)
                line_num += 1
        # number of lines must match number of columns
        return line_num == self._size

    def check(self) -> (int, int):
        """
        :return: Number of mistakes found, ratio
        """
        mistakes = 0

        # check rows
        for row in self._board:
            if not self._check_unique(row):
                mistakes += 1
        # check columns
        for i in range(self._size):
            col = self._board[:, i]
            if not self._check_unique(col):
                mistakes += 1
        # check squares
        for square in self.squares():
            if not self._check_unique(square):
                mistakes += 1

        return mistakes, mistakes / (3 * self._size)

    def squares(self, flatten: bool = True):
        """
        Generator, produces all square constraint areas on the board
        :type flatten: Flattens the square to 1D array when set to True
        :return: Iterator to all square areas
        """
        # horizontal
        for hr in range(self._size_sq):
            # vertical
            for vert in range(self._size_sq):
                # count area constraints
                area_hr = hr * self._size_sq, (hr + 1) * self._size_sq
                area_vert = vert * self._size_sq, (vert + 1) * self._size_sq
                # cut a square from the board & check it
                square = self._board[area_vert[0]:area_vert[1], area_hr[0]:area_hr[1]]
                # yield the square
                if flatten:
                    yield square.flatten()
                else:
                    yield square

    def _offset_number(self, num: int) -> str:
        """
        :return: Board-supported printable version of the number
        """
        rest = self._max_digits - self._count_digits(num)
        return ' ' * ((rest + 1) // 2) + str(num) + ' ' * (rest // 2)

    def _assemble_path(self):
        """
        :return: Path to valid sample data
        """
        return _SAMPLE_INIT_PATH + str(self._size) + _SAMPLE_INIT_FORMAT

    @staticmethod
    def _check_unique(area):
        """
        Checks area (set of of tiles) - lines, columns or squares
        :return: True when the are is ok according to sudoku rules, otherwise False
        """
        # possibilities
        free = set(range(1, len(area) + 1))
        # go through all tiles, check for conflicts or unfilled tiles
        for x in area:
            if x == 0 or x not in free:
                # x is invalid
                return False
            # x is valid, remove it from viable options
            free.remove(x)
        return True

    @staticmethod
    def _count_digits(num: int) -> int:
        """
        Counts number of digits in the number
        :param num: number
        :return: number of digits
        """
        # edge case
        if num == 0:
            return 1
        # rest
        rv = 0
        while num > 0:
            num //= 10
            rv += 1
        return rv



