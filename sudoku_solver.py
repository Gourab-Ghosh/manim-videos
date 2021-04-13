from manim import *
from copy import deepcopy

sudoku = np.array([
    [0, 9, 4,    0, 3, 0,    1, 0, 0],
    [8, 1, 2,    7, 0, 0,    0, 9, 6],
    [3, 0, 0,    1, 9, 0,    0, 0, 0],

    [0, 3, 0,    9, 0, 4,    6, 0, 0],
    [0, 0, 8,    6, 1, 3,    0, 4, 9],
    [0, 0, 6,    2, 0, 0,    0, 0, 1],

    [4, 0, 3,    5, 0, 0,    0, 0, 8],
    [5, 0, 0,    0, 2, 0,    7, 0, 0],
    [0, 6, 0,    0, 0, 8,    4, 1, 5],
])

def coord(x, y, z = 0):
    return np.array([x, y, z])

def do_nothing(*args, **kwargs):
    pass

class SudokuMobject(VGroup):

    CFG = {
        "sudoku_width": config.frame_height - 0.5,
        "sudoku_height": config.frame_height - 0.5,
        # "rectangle_color": GREEN_E,
        "rectangle_color": RED,
        "thick_grid_stroke_width": 8,
        "thick_grid_color": RED,
        "thin_grid_stroke_width": 2,
        "thin_grid_color": RED,
        "integer_class": Integer,
        # "integer_class": MathTex,
        "digits_scale_factor_buff": 0.35,
        # "digits_scale_factor_buff": 0,

        "integer_kwargs": {
            "color": WHITE,
        },

        "selected_index_rectangle_kwargs": {
            "color": ORANGE,
            "stroke_width": 0,
            "fill_opacity": 1,
        },
    }

    def __init__(
            self,
            sudoku_array,
            scene = None,
            num_row_divisions = None,
            num_coloumn_divisions = None,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.__dict__.update(self.CFG)

        if type(sudoku_array) is np.ndarray:
            self.sudoku_array = sudoku_array
        else:
            self.sudoku_array = np.array(sudoku_array)
        self.scene = scene

        self.num_row_divisions = num_row_divisions
        self.num_column_divisions = num_coloumn_divisions

        self.define_required_variables()
        self.generate_grids()
        self.fill_numbers()

    def define_required_variables(self):
        self.num_rows, self.num_columns = self.sudoku_array.shape

        if self.num_row_divisions is None:
            self.num_row_divisions = int(round(np.sqrt(self.num_rows)))
        if self.num_column_divisions is None:
            self.num_column_divisions = int(round(np.sqrt(self.num_columns)))

        self.filled_spaces = {}

        self._selected_index = None
        self.selected_index_rectangle = Rectangle(
            height = self.sudoku_height / self.num_rows,
            width = self.sudoku_width / self.num_columns,
            **self.selected_index_rectangle_kwargs,
        )

    def generate_grids(self):
        self.border = Rectangle(
            width = self.sudoku_width,
            height = self.sudoku_height,
            stroke_width = self.thick_grid_stroke_width,
            color = self.rectangle_color,
        )

        self.add(self.border)

        self.thin_grids = VGroup(*[
            Line(
                self.border.point_from_proportion(start_proportion),
                self.border.point_from_proportion(end_proportion),
                stroke_width = self.thin_grid_stroke_width,
                color = self.thick_grid_color,
            )
            for start_proportion, end_proportion in zip(
                [1 - i/(4 * self.num_rows) for i in range(1, self.num_rows)] + [i/(4 * self.num_columns) for i in range(1, self.num_columns)],
                [0.25 + i/(4 * self.num_rows) for i in range(1, self.num_rows)] + [0.75 - i/(4 * self.num_columns) for i in range(1, self.num_columns)],
            )
        ])

        self.add(self.thin_grids)

        self.thick_grid_indices = []
        self.thick_grid_indices += list(np.round(np.linspace(-1, self.num_rows-1, self.num_row_divisions+1))[1:-1])
        self.thick_grid_indices += list(np.round(np.linspace(-1, self.num_columns-1, self.num_column_divisions+1))[1:-1] + self.num_rows - 1)
        self.thick_grid_indices = [int(i) for i in self.thick_grid_indices]

        for i in self.thick_grid_indices:
            self.thin_grids[i].set_color(color = self.thin_grid_color)
            self.thin_grids[i].set_stroke(width = self.thick_grid_stroke_width)

    def fill_numbers(self):
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                value = self.sudoku_array[i, j]
                if value != 0:
                    self.modify_grid([i, j], value)

    def adjust_integer_size(self, mobject):
        mobject_initial_height = mobject.height
        mobject_final_height = max(0, self.sudoku_height / self.num_rows - self.digits_scale_factor_buff)

        mobject_initial_width = mobject.width
        mobject_final_width = max(0, self.sudoku_width / self.num_columns - self.digits_scale_factor_buff)

        scale_facor = min(
            mobject_final_height / mobject_initial_height,
            mobject_final_width / mobject_initial_width,
        )

        mobject.scale(scale_facor)

        return mobject

    def get_center_from_grid_index(self, grid_index):
        coordinate = coord(*[
                location_coord + multiplier * (index2 + 1/2) * (length / num_divisions)
                for location_coord, multiplier, index2, length, num_divisions in zip(
                    self.border.point_from_proportion(0),
                    [1, -1],
                    grid_index[::-1],
                    [self.sudoku_width, self.sudoku_height],
                    [self.num_columns, self.num_rows],
                )
            ])

        return coordinate

    def modify_grid(self, grid_index, value, color = None):
        grid_index = tuple(grid_index)

        try:
            mobject = self.filled_spaces[grid_index]
        except KeyError:
            mobject = self.adjust_integer_size(self.integer_class(value, **self.integer_kwargs))
            self.filled_spaces[grid_index] = mobject
            coordinate = self.get_center_from_grid_index(grid_index)
            mobject.move_to(coordinate)
            self.add(mobject)
        else:
            if value == 0:
                self.remove(mobject)
            else:
                self.add(mobject)
                if isinstance(mobject, DecimalNumber):
                    mobject.set_value(value)
                else:
                    mobject.become(
                        self.integer_class(value, **self.integer_kwargs)
                    )
                self.adjust_integer_size(mobject)
        finally:
            if color is not None:
                mobject.set_color(color)

    @property
    def selected_index(self):
        return self._selected_index

    @selected_index.setter
    def selected_index(self, grid_index):
        if grid_index is None:
            self.remove(self.selected_index_rectangle)
        else:
            grid_index = tuple(grid_index)
            if self._selected_index is None:
                self.add_to_back(self.selected_index_rectangle)

            coordinate = self.get_center_from_grid_index(grid_index)
            self.selected_index_rectangle.move_to(coordinate)

        self._selected_index = grid_index

    def select_index(self, index):
        self.selected_index = index

    def scale(self, scale_factor):
        attrs = [
            "sudoku_width",
            "sudoku_height",
            "digits_scale_factor_buff",
        ]

        for attr in attrs:
            setattr(
                self,
                attr,
                scale_factor * getattr(self, attr)
            )

        super().scale(scale_factor)

class SolveSudoku(Animation):

    CFG = {
        "rate_func": linear,
    }

    def __init__(self, sudoku, **kwargs):
        self.__dict__.update(self.CFG)

        super().__init__(
            sudoku,
            rate_func = self.rate_func,
            **kwargs
        )

        self.define_important_constants()

        self.mobject.rect =  Rectangle(
            height = self.mobject.sudoku_height / self.mobject.num_rows,
            width = self.mobject.sudoku_width / self.mobject.num_columns,
            **{"color": YELLOW,"stroke_width": 0,"fill_opacity": 1,},
        )

        self.mobject.add_to_back(self.mobject.rect)

        sudoku_array = deepcopy(self.mobject.sudoku_array)
        self.solve_sudoku(sudoku_array)

    def define_important_constants(self):
        self.list_of_actions = []
        self.last_action_index = -1
        self.num_rows, self.num_columns = self.mobject.sudoku_array.shape
        self.thick_grid_row_indices = [0] + [x + 1 for x in self.mobject.thick_grid_indices if x < self.num_rows] + [self.num_rows]
        self.thick_grid_column_indices = [0] + [x - self.num_rows + 2 for x in self.mobject.thick_grid_indices if x >= self.num_rows] + [self.num_columns]
        self.max_possible_value = max(self.num_rows, self.num_columns)

    def solve_sudoku(self, sudoku_array):

        def get_bounds(index):
            i, j = index

            row_lower_bound = max(filter(
                lambda x: x <= i,
                self.thick_grid_row_indices,
            ))

            column_lower_bound = max(filter(
                lambda x: x <= j,
                self.thick_grid_column_indices,
            ))

            return row_lower_bound, column_lower_bound

        def is_correct_value(index, value):
            nonlocal sudoku_array
            index = tuple(index)
            i, j = index

            for x in range(self.num_rows):
                self.list_of_actions.append([self.mobject.select_index, [(x, j)]])
                if sudoku_array[x, j] == value and x != i:
                    return False

            for y in range(self.num_columns):
                self.list_of_actions.append([self.mobject.select_index, [(i, y)]])
                if sudoku_array[i, y] == value and y != j:
                    return False

            row_lower_bound, column_lower_bound = get_bounds(index)
            next_row_lower_bound = min(filter(lambda x: x > row_lower_bound, self.thick_grid_row_indices))
            next_column_lower_bound = min(filter(lambda x: x > column_lower_bound, self.thick_grid_column_indices))

            n = 0

            for x in range(row_lower_bound, next_row_lower_bound):
                for y in range(column_lower_bound, next_column_lower_bound):
                    self.list_of_actions.append([self.mobject.select_index, [(x, y)]])
                    n += 1
                    if sudoku_array[x, y] == value and (x, y) != index:
                        return False

            self.max_possible_value = max(self.max_possible_value, n)

            return True

        def get_empty_space():
            nonlocal sudoku_array
            self.list_of_actions.append([self.mobject.selected_index_rectangle.set_color, [BLUE]])
            for i in range(self.num_rows):
                for j in range(self.num_columns):
                    self.list_of_actions.append([self.mobject.select_index, [(i, j)]])
                    if sudoku_array[i, j] == 0:
                        self.list_of_actions.append([self.mobject.selected_index_rectangle.set_color, [ORANGE]])
                        return (i, j)

        rect = self.mobject.rect
        index = get_empty_space()

        if not index:
            return True
        else:
            self.list_of_actions.append([rect.move_to, [self.mobject.get_center_from_grid_index(index)]])
            current_val = 0
            while current_val < self.max_possible_value:
                current_val += 1
                self.list_of_actions.append([self.mobject.modify_grid, [index, current_val, BLUE]])
                if is_correct_value(index, current_val):
                    sudoku_array[index] = current_val
                    self.list_of_actions.append([self.mobject.modify_grid, [index, current_val, GREEN]])

                    if self.solve_sudoku(sudoku_array):
                        return True

                    sudoku_array[index] = 0
                    self.list_of_actions.append([rect.move_to, [self.mobject.get_center_from_grid_index(index)]])
                    self.list_of_actions.append([self.mobject.modify_grid, [index, 0, GREEN]])

            self.list_of_actions.append([self.mobject.modify_grid, [index, 0, GREEN]])
            return False

    def begin(self):
        self.list_of_actions = [
            i + [do_nothing, [], {}][len(i):]
            for i in self.list_of_actions
        ]

        super().begin()

    def interpolate_mobject(self, alpha):
        last_action_index = int(round(interpolate(
            0,
            len(self.list_of_actions) - 1,
            alpha,
        )))

        for i in range(self.last_action_index + 1, last_action_index + 1):
            func, args, kwargs = self.list_of_actions[i]
            func(*args, **kwargs)

        self.last_action_index = last_action_index

    def finish(self):
        super().finish()
        self.mobject.remove(self.mobject.selected_index_rectangle, self.mobject.rect)

class Testing(Scene):

    def construct(self):
        s = SudokuMobject(sudoku)
        self.play(Write(s))
        self.wait(2)
        self.play(SolveSudoku(s), run_time = 5)
        self.wait(5)