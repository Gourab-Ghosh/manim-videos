from manim import *
from copy import deepcopy
import random
# import numpy as np
# from numpy import exp, sin, cos, tan, arcsin, arccos, arctan, sqrt, abs, sign

# Mobject

def digest_config(self, kwargs):
    for cls in reversed(self.__class__.mro()):
        if hasattr(cls, "CFG"):
            if self.CFG != {}:
                self.__dict__.update(cls.CFG)
    if kwargs != {}:
        self.__dict__.update(kwargs)

def coord(x, y, z = 0):
	return np.array([x,y,z])

roundint = lambda x: int(round(x))

def choose_objects_randomly(*args, **kwargs):
	# Enter (object, probability) as tuple

	object_list = []
	probability_list = []

	for object_tuple in args:
		object_list.append(object_tuple[0])
		probability_list.append(object_tuple[1]) 

	probability_list_sum = sum(probability_list)
	probability_list = [probability/probability_list_sum for probability in probability_list]
	random_number = random.uniform(0, 1)
	cumulative_probability_list = [0]

	for probability in probability_list:
		cumulative_probability_list.append(cumulative_probability_list[-1] + probability)

	for i in range(len(probability_list)):
		if cumulative_probability_list[i] < random_number < cumulative_probability_list[i+1]:
			obj = object_list[i]
			break
	try:
		obj
	except NameError:
		obj = choose_objects_randomly(*args, **kwargs)
	return obj

class MazeGridMobject(Rectangle):
	CFG = {
	"fill_opacity" : 0.5,
	"background_stroke_width" : 0,
	"background_stroke_opacity" : 0,
	"stroke_width" : 0,
	"colors" : {
		"before_visiting_color" : BLUE,
		"during_visit_color" : PURPLE,
		"after_visiting_color" : ORANGE,
		},
	}

	def __init__(self, state, **kwargs):
		digest_config(self, kwargs)
		super().__init__(
			fill_opacity = self.fill_opacity,
			background_stroke_width = self.background_stroke_width,
			background_stroke_opacity = self.background_stroke_opacity,
			stroke_width = self.stroke_width,
			**kwargs,
		)
		self._state = state
		self.update_color()

	@property
	def state(self):
		return self._state

	@state.setter
	def state(self, new_state):
		self._state = new_state
		self.update_color()

	def update_color(self):
		if self.state == "unvisited":
			self.set_color(self.colors["before_visiting_color"])
		if self.state == "active":
			self.set_color(self.colors["during_visit_color"])
		if self.state == "visited":
			self.set_color(self.colors["after_visiting_color"])

class MazeGridBorderMobject(Line):
	CFG = {
	"color" : WHITE,
	}

	def __init__(self, *args, state = "visible", **kwargs):
		digest_config(self, kwargs)
		super().__init__(*args, **kwargs)
		self._state = state
		self.update_color()

	@property
	def state(self):
		return self._state

	@state.setter
	def state(self, new_state):
		self._state = new_state
		self.update_color()

	def update_color(self):
		if self.state == "visible":
			self.set_stroke(self.color, opacity = 1)
		if self.state == "invisible":
			self.set_stroke(self.color, opacity = 0)

class MazeMobject(VMobject):
	CFG = {
	"maze_width" : config.frame_height - 1,
	"maze_height" : config.frame_height - 1,
	}

	def __init__(self, **kwargs):
		digest_config(self, kwargs)
		super().__init__(**kwargs)
		self.calculate_required_items()
		self.create_required_objects()
		self.get_grid(self.starting_grid_index).state = "active"
		self.current_active_grid_index = self.starting_grid_index

	def calculate_required_items(self):
		self.grid_width = self.maze_width / self.num_columns
		self.grid_height = self.maze_height / self.num_rows
		self.maze_corners = [
			coord(
				i*self.maze_width/2,
				j*self.maze_height/2,
				)
				for i, j in [(-1,1), (1,1), (1,-1), (-1,-1)]
			]
		self.grid_corners = self.get_grid_corners()

	def get_grid_corners(self): # done
		grid_corner_coordinates = np.zeros(
			(
				self.num_rows + 1,
				self.num_columns + 1
				),
			dtype = np.ndarray
			)
		top_row_grid_coordinates = np.linspace(
			*self.maze_corners[:2],
			self.num_columns + 1
			)
		all_columns = [
			np.linspace(
				coordinate,
				coordinate - coord(0, self.maze_height),
				self.num_rows+1
				) for coordinate in top_row_grid_coordinates
			]
		for i, column in enumerate(all_columns):
			for j in range(self.num_rows + 1):
				grid_corner_coordinates[j, i] = column[j]
		return grid_corner_coordinates

	def create_required_objects(self): # done
		self.grid = self.create_grids()
		self.grid_borders = VGroup(*self.create_grid_borders())
		self.add(self.grid, self.grid_borders)

	def create_grids(self): # done
		grid_mob = VGroup()
		self.grid_array = np.zeros(
			(
				self.num_rows + 1,
				self.num_columns + 1
				),
			dtype = MazeGridMobject
			)
		for i in range(self.num_rows):
			for j in range(self.num_columns):
				grid_coordinate = (self.grid_corners[i, j] + self.grid_corners[i+1, j+1]) / 2
				grid = self.create_grid_with_required_properties(grid_coordinate)
				self.grid_array[i, j] = grid
				grid_mob.add(grid)
		return grid_mob

	def create_grid_with_required_properties(self, grid_coordinate): # done
		grid = MazeGridMobject(
			state = "unvisited",
			width = self.grid_width,
			height = self.grid_height
			)
		grid.move_to(grid_coordinate)

		return grid

	def create_grid_borders(self): # done
		self.grid_borders_along_x = VGroup()
		self.grid_borders_along_y = VGroup()

		for row in self.grid_corners:
			for i in range(self.num_columns):
				self.grid_borders_along_x.add(
					MazeGridBorderMobject(
						row[i],
						row[i+1],
						)
					)

		for column in self.grid_corners.T:
			for i in range(self.num_rows):
				self.grid_borders_along_y.add(
					MazeGridBorderMobject(
						column[i],
						column[i+1],
						)
					)

		return self.grid_borders_along_x, self.grid_borders_along_y

	def get_grid(self, grid_index): # done
		x, y = grid_index
		return self.grid_array[roundint(y), roundint(x)]

	def change_grid_state(self, grid_index, state): # done
		self.get_grid(grid_index).state = state

	def get_common_side(self, grid1_index, grid2_index): # done
		if abs(grid1_index[0] - grid2_index[0]) == 0 and abs(grid1_index[1] - grid2_index[1]) == 1:
			req_y_index = max(grid1_index[1], grid2_index[1])
			return self.grid_borders_along_x[self.num_columns * req_y_index + grid1_index[0]]
		elif abs(grid1_index[0] - grid2_index[0]) == 1 and abs(grid1_index[1] - grid2_index[1]) == 0:
			req_x_index = max(grid1_index[0], grid2_index[0])
			return self.grid_borders_along_y[self.num_rows * req_x_index + grid2_index[1]]
		else:
			raise Exception(f"The grids with grid index {grid1_index} and {grid2_index} has no common side.")

	def get_adjacent_grid_index(self, current_grid, direction):
		return [
			roundint(
				current_grid[0] + direction[0]
				),
			roundint(
				current_grid[1] - direction[1]
				),
			]

	def change_active_grid(self, current_active_grid_index, new_active_grid_index): # done
		self.change_grid_state(current_active_grid_index, "visited")
		self.change_grid_state(new_active_grid_index, "active")
		self.get_common_side(current_active_grid_index, new_active_grid_index).state = "invisible"
		self.current_active_grid_index = new_active_grid_index

	def change_active_grid_to(self, new_active_grid_index): # done
		self.change_active_grid(self.current_active_grid_index, new_active_grid_index)

class RandomMazeCreator(MazeMobject):
	CFG = {
	"num_rows" : 10,
	"num_columns" : 10,
	"starting_grid_index" : [0, 0],
	"tracing_path" : [],
	}
	def __init__(self, scene, **kwargs):
		digest_config(self, kwargs)
		super().__init__(**kwargs)
		self.scene = scene
		if hasattr(self, "seed"):
			random.seed(self.seed)

	def wait(self, *args, **kwargs):
		self.scene.wait(*args, **kwargs)

	def play(self, *args, **kwargs):
		self.scene.play(*args, **kwargs)

	def get_possible_directions(self):
		directions = [list(direction) for direction in (UP, RIGHT, DOWN, LEFT)]
		x, y = self.current_active_grid_index[:2]

		if x == 0:
			directions.pop(
				directions.index(
					list(
						LEFT
						)
					)
				)
		if x == self.num_columns - 1:
			directions.pop(
				directions.index(
					list(
						RIGHT
						)
					)
				)
		if y == 0:
			directions.pop(
				directions.index(
					list(
						UP
						)
					)
				)
		if y == self.num_rows - 1:
			directions.pop(
				directions.index(
					list(
						DOWN
						)
					)
				)
		for direction in deepcopy(directions):
			grid_state = self.get_grid(
				self.get_adjacent_grid_index(
					direction
					)
				).state

			if grid_state == "visited":
				directions.pop(
					directions.index(
						list(
							direction
							)
						)
					)
		# if len(directions) > 1 and hasattr(self, "last_direction"):
		# 	for direction in deepcopy(directions):
		# 		if direction == self.last_direction:
		# 			directions.pop(
		# 				directions.index(
		# 					list(
		# 						direction
		# 						)
		# 					)
		# 				)

		return directions

	def choose_direction_randomly(self):
		directions = self.get_possible_directions()

		if len(directions) == 0:
			return

		probabilities = [1/len(directions) for _ in directions]

		chosen_direction = coord(
			*choose_objects_randomly(
				*zip(
					directions,
					probabilities,
					)
				)
			)

		self.last_direction = list(chosen_direction)

		return chosen_direction

	def change_active_grid_randomly(self, animate = True):
		chosen_direction = self.choose_direction_randomly()

		if type(chosen_direction) == np.ndarray:
			new_active_grid_index = self.get_adjacent_grid_index(chosen_direction)
			self.tracing_path.append(self.current_active_grid_index)
		else:
			new_active_grid_index = self.tracing_path.pop(-1)

		if animate:
			self.wait(0.1)
			pass
		self.change_active_grid_to(new_active_grid_index)

	def make_paths(self, tolerance = np.inf, animate = True):
		n = 0
		tolerance -= 1
		# tolerance = 2 * self.num_rows * self.num_columns
		self.change_active_grid_randomly(animate)
		while self.current_active_grid_index != self.starting_grid_index and n <= tolerance:
			self.change_active_grid_randomly(animate)
			n += 1

	def get_adjacent_grid_index(self, direction):
		return super().get_adjacent_grid_index(self.current_active_grid_index, direction)

class MazeSolver:
	def __init__(self, **kwargs):
		digest_config(self, kwargs)
		pass

class Testing(Scene):
	def construct(self):
		maze = RandomMazeCreator(self, num_rows = 20, num_columns = 20, seed = 10)
		self.add(maze)
		maze.grid_borders_along_y[0].state = "invisible"
		maze.grid_borders_along_y[-1].state = "invisible"
		# maze.make_paths(animate = False)
		maze.make_paths(animate = True)
		maze.get_grid([0, 0]).state = "visited"
		self.wait(5)