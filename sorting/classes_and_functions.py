from manim import *
from manim.utils.bezier import interpolate, inverse_interpolate, match_interpolate
from tqdm import tqdm
from copy import deepcopy
import os, platform, hashlib, wave, struct, contextlib, itertools as it

def digest_config(self, kwargs):
    for cls in reversed(self.__class__.mro()):
        if hasattr(cls, "_CONFIG"):
            if self._CONFIG != {}:
                self.__dict__.update(cls._CONFIG)
    if kwargs != {}:
        self.__dict__.update(kwargs)

MEDIA_DIR = "media"

SORTING_ALGORITHMS = []

def sorting_algorithm(sort_class):
    global SORTING_ALGORITHMS

    if hasattr(sort_class, "sort"):
        SORTING_ALGORITHMS.append(sort_class)

    return sort_class

def coord(x, y, z = 0):
    return np.array([x, y, z])

def do_nothing(*args, **kwargs):
    pass

def interpolate_color(colors, value, min_value, max_value):
    num_colors = len(colors)

    if num_colors == 0:
        return
    if num_colors == 1:
        return colors[0]

    colors = list(
        map(
            color_to_rgb,
            colors,
        )
    )

    if min_value == max_value:
        return rgb_to_color(colors[0])

    ratio = inverse_interpolate(min_value, max_value, value)
    ranges = np.linspace(0, 1, num_colors)

    for i in range(num_colors):
        if ranges[i] <= ratio <= ranges[i+1]:
            selected_range = ranges[i:i+2]
            selected_colors = colors[i:i+2]
            break

    rgb = match_interpolate(*selected_colors, *selected_range, ratio)

    return rgb_to_color(rgb)

def swap(given_list, index1, index2):
    given_list[index1], given_list[index2] = given_list[index2], given_list[index1]

def get_vector_length(vector):
    return np.sqrt(
        (
            vector ** 2
        ).sum()
    )

def get_line_length(line):
    return get_vector_length(
        line.find_line_end() - line.find_line_start()
    )

def exponential(t, exp_factor = 5):
    return np.clip(match_interpolate(
        0,
        1,
        1,
        np.exp(exp_factor),
        np.exp(exp_factor * t),
    ), 0, 1)

class CheckErrorInCodes:
    def __init__(self, num_iter = 100, num_data = 100):
        self.num_iter = num_iter
        self.num_data = num_data

        self.test_for_correct_code()

    def test_for_correct_code(self):
        global SORTING_ALGORITHMS
        sorting_algorithms = deepcopy(SORTING_ALGORITHMS)
        num_sorts = len(sorting_algorithms)
        sorting_algorithms = it.cycle(sorting_algorithms)
        num_testing = 0

        for _ in range(self.num_iter):
            num_testing += 1
            print(
                "testing {}".format(
                    num_testing
                )
            )

            data = np.random.random(self.num_data)
            for _ in range(num_sorts):
                sort = next(sorting_algorithms)
                sort_algo = sort(data)

class AudioFileGenerator:

    _CONFIG = {
        "starting_frequency": 120,
        "ending_frequency": 1212,
        "sample_rate": 44100,
        "nchannels": 1,
        "sampwidth": 2,
        "comptype": "NONE",
        "compname": "not compressed",
        "min_volume": 1,
        "max_volume": 1,
        "max_num_beats_per_sec": 20,
        "max_num_beats_overlap": 3,
        "run_time_ratio_of_smoothing_curve": 0.1,
        "lag_ratio_rate_func": lambda t: exponential(t, 5),

        "use_old_files": True,
        "file_path": os.path.join(MEDIA_DIR, "sounds"),
        "file_extension": "wav",
    }

    def __init__(self, frequencies, run_time, leave_progress_bars, **kwargs):
        digest_config(self, kwargs)

        self.frequencies = list(frequencies)
        self.leave_progress_bars = leave_progress_bars
        self.run_time = run_time

        self.useless_settings = ["use_old_files", "file_path", "file_extension"]

        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)

        self.define_important_constants()

        if self.max_num_beats_per_sec:
            self.analyse_and_fix_frequencies()

        self.generate_file_name()

    def calculate_lag_ratio(self):
        upper_bound = 50 * self.run_time

        return self.lag_ratio_rate_func(
            match_interpolate(
                1,
                0,
                0,
                upper_bound,
                np.clip(len(self.frequencies), 0, upper_bound),
            )
        )

    def define_important_constants(self):
        self.smoothing_curve_run_time = self.run_time_ratio_of_smoothing_curve * self.run_time / len(self.frequencies)

        if not hasattr(self, "lag_ratio"):
            self.lag_ratio =  self.calculate_lag_ratio()

    def analyse_and_fix_frequencies(self):
        arbitary_frequencies = []
        run_time_per_beat = max(1 / self.max_num_beats_per_sec, self.run_time / len(self.frequencies))
        self.run_time_per_beat = run_time_per_beat
        run_time_per_beat /= interpolate(self.max_num_beats_overlap, 1, self.lag_ratio)
        expected_num_frequencies = int(round(self.run_time / run_time_per_beat))
        current_num_frequencies = len(self.frequencies)

        indices_to_select = np.linspace(
            0,
            current_num_frequencies - 1,
            expected_num_frequencies,
            dtype = int,
        )

        for i in range(current_num_frequencies):
            if i in indices_to_select:
                arbitary_frequencies.append(self.frequencies[i])

        self.frequencies = arbitary_frequencies

    def get_settings(self):
        possible_settings = sorted(list(AudioFileGenerator._CONFIG) + ["lag_ratio", "run_time"])

        for i in self.useless_settings:
            possible_settings.remove(i)

        settings = [self.__dict__.get(i) for i in possible_settings]

        return settings

    def get_sinewave(self, frequency, duration, volume):
        num_samples = int(duration * self.sample_rate)
        partial_audio = np.linspace(0, num_samples - 1, num_samples)
        partial_audio = volume * np.sin(2 * PI * frequency * (partial_audio / self.sample_rate))

        return partial_audio

    def get_volume_form_frequency(self, frequency):
        volume = match_interpolate(
                self.min_volume,
                self.max_volume,
                self.starting_frequency,
                self.ending_frequency,
                np.clip(
                    frequency,
                    self.starting_frequency,
                    self.ending_frequency,
                ),
            )

        return volume

    def generate_sinewave_from_frequency_list(self, frequency_list, duration):
        sinewave = np.zeros(int(duration * self.sample_rate))

        sinewave_func = lambda frequency: self.get_sinewave(
            np.clip(
                frequency,
                self.starting_frequency,
                self.ending_frequency,
            ),
            duration,
            self.get_volume_form_frequency(frequency),
        )

        for frequency in frequency_list:
            sinewave += sinewave_func(frequency)
        if len(frequency_list) != 0:
            sinewave /= len(frequency_list)

        return sinewave

    def get_current_run_time_from_audio(self, audio):
        current_run_time = len(audio) / self.sample_rate
        return current_run_time

    def get_expected_current_run_time(self, num_iters_passed):
        expected_current_run_time = num_iters_passed * self.run_time / len(self.frequencies)
        return expected_current_run_time

    def get_index_in_audio_from_time(self, time):
        index = int(round(self.sample_rate * time))
        return index

    def merge_sinewave(self, audio, sinewave, merge_starting_time):
        starting_index = np.clip(
            self.get_index_in_audio_from_time(merge_starting_time),
            0,
            len(audio) - 1,
        )

        for i in range(len(sinewave)):
            current_index = i + starting_index

            if 0 <= current_index <= len(audio) - 1:
                audio[current_index] = np.average([audio[current_index], sinewave[i]])
            else:
                audio.append(sinewave[i])

    def smoothen_audio(self, audio):
        from collections import defaultdict
        d = defaultdict(int)
        diff = np.round(np.abs(np.diff(audio)), 2)
        diff.sort()
        for i in diff:
            d[i] += 1
        print(d)
        # for i in range(len(audio) - 2):
        #   if not np.allclose(audio[i], audio[i+1], audio[i+2]):
        #       a.append(i)
        # print(a)

    def get_full_audio(self):
        audio = []

        for i, frequency_list in enumerate(self.frequencies):
            sinewave = self.generate_sinewave_from_frequency_list(frequency_list, self.run_time_per_beat)
            expected_current_run_time = self.get_expected_current_run_time(i+1)
            self.merge_sinewave(audio, sinewave, expected_current_run_time)
            while self.get_current_run_time_from_audio(audio) < expected_current_run_time:
                audio.append(audio[-1])

        # self.smoothen_audio(audio)

        return audio

    def generate_file_name(self):
        file_name = str(self.get_settings()) + str(self.frequencies)

        hasher = hashlib.sha256()
        hasher.update(file_name.encode())

        if not self.file_extension.startswith("."):
            self.file_extension = "." + self.file_extension

        self.file_name = "".join([hasher.hexdigest()[:16], self.file_extension])
        self.file_name_with_path = os.path.abspath(self.file_path + os.sep + self.file_name)

    def perfect_file_generated(self, tolerance = 1e-1):
        if os.path.isfile(self.file_name_with_path):
            with contextlib.closing(wave.open(self.file_name_with_path, "r")) as file:
                frames = file.getnframes()
                rate = file.getframerate()
                duration = frames / float(rate)
            return np.abs(duration - self.run_time) <= tolerance
        return False

    def generate_file(self):
        if not self.perfect_file_generated():
            self.use_old_files = False

        if not self.use_old_files:
            audio = self.get_full_audio()
            nframes = len(audio)

            if os.path.isfile(self.file_name_with_path):
                os.remove(self.file_name_with_path)

            with wave.open(self.file_name_with_path, "w") as wav_file:
                wav_file.setparams((
                    self.nchannels,
                    self.sampwidth,
                    self.sample_rate,
                    nframes,
                    self.comptype,
                    self.compname,
                ))

                for sample in tqdm(
                    audio,
                    desc = "Writing audio to " + self.file_name,
                    leave = self.leave_progress_bars,
                    ascii = False if platform.system() != 'Windows' else True,
                ):
                    wav_file.writeframes(
                        struct.pack(
                            "h",
                            int(sample * 32767)
                        )
                    )

    def get_file_name(self):
        return self.file_name_with_path

class AudioFileGeneratorTrial(AudioFileGenerator):
    pass

class ModifiedLine1(Line):
    def __init__(self, start, end, interpolate_color_func, **kwargs):
        digest_config(self, kwargs)
        super().__init__(
            start,
            end,
            **kwargs,
        )

        self.interpolate_color_func = interpolate_color_func
        self.set_color(self.interpolate_color_func(self))

    def find_line_start(self):
        return self.get_start()

    def find_line_end(self):
        return self.get_end()

    def put_start_and_end_on(self, start, end):
        if (start == end).all():
            end += 1e-6 * UP
        super().put_start_and_end_on(start, end)
        self.set_color(self.interpolate_color_func(self))

class ModifiedLine2(Rectangle):

    def __init__(self, interpolate_color_func, **kwargs):
        super().__init__(
            **kwargs,
            stroke_width = 0,
        )
        digest_config(self, kwargs)

        self.interpolate_color_func = interpolate_color_func
        self.set_color(self.interpolate_color_func(self))
        self.set_fill(opacity = 1)

    def put_start_and_end_on(self, start, end):
        center = (start + end) / 2
        length = get_vector_length(end - start)

        self.stretch_to_fit_height(
            max(
                1e-6,
                length,
            )
        )

        self.move_to(center)
        self.set_color(self.interpolate_color_func(self))

        return self

    def find_line_start(self):
        return self.point_from_proportion(0.625)

    def find_line_end(self):
        return self.point_from_proportion(0.125)

class Data(VGroup):

    _CONFIG = {
        "data_height": config.frame_height - 0.5,
        "data_width": config.frame_width - 0.5,
        "colors": ["#0000AA", "#00AA00", "#AA0000"],
        "selected_line_color": [WHITE],
        "partial_colors": ["#7D0D82", "#690F96"],
        "gap_between_lines": 0.04,
        "gap_in_sides": 0.1,
        "gap_from_base_line": 0.05,
        "rect_min_width_tolerance": 0.02,
        "base_stroke_width": 5,
        "stroke_width_constant": 2,
    }

    def __init__(self, data_list, **kwargs):
        super().__init__(**kwargs)
        digest_config(self, kwargs)
        self.data_list = list(data_list)

        self._create_initial_vars()
        self._create_base()

        self.lines = self._get_lines()
        self._update_line_colors()

        self.add(self.lines)
        self.add(self.base)

    def _create_initial_vars(self):
        self.num_data = len(self.data_list)
        self.initial_data_list = deepcopy(self.data_list)
        self._prev_data_list = deepcopy(self.initial_data_list)
        self.initial_min = min(self.data_list)
        self.initial_max = max(self.data_list)
        self.update_line_indices = []
        self.num_comparisons_done = 0
        self.num_values_changed = 0

    def _create_base(self):
        self.base_line_coordinates = [
            coord(
                -self.data_width / 2,
                -self.data_height / 2,
            ),
            coord(
                self.data_width / 2,
                -self.data_height / 2,
            ),
        ]

        self.base = Line(*self.base_line_coordinates, stroke_width = self.base_stroke_width)

    def _set_line_colour(self, line):
        min_line_height = self._calculate_line_height(self.initial_min)

        line_height = np.clip(
            get_vector_length(
                line.find_line_end() - line.find_line_start(),
            ),
            min_line_height,
            self.data_height,
        )

        line.set_color(
            interpolate_color(
                self.colors,
                line_height,
                min_line_height,
                self.data_height,
            )
        )

        line.to_change_color = False

    def _calculate_line_height(self, value):
        return value / self.initial_max * self.data_height

    def _calculate_line_width(self):
        return max(0, self.data_width / len(self.data_list) - self.gap_between_lines)

    def _get_lines(self):
        line_width = self._calculate_line_width()

        line_base_coordinates = np.linspace(
            self.base.get_start() + coord(self.gap_in_sides + line_width / 2, self.gap_from_base_line),
            self.base.get_end() - coord(self.gap_in_sides + line_width / 2, -self.gap_from_base_line),
            self.num_data,
        )

        line_upper_coordinates = [
            line_base_coordinates[i] + coord(0, self._calculate_line_height(self.initial_data_list[i]))
            for i in range(self.num_data)
        ]

        if line_width < self.rect_min_width_tolerance:
            line_stroke_width = self.stroke_width_constant * 1000 / self.num_data

            line_group = VGroup(*[
                ModifiedLine1(
                    *end_points,
                    self._set_line_colour,
                    stroke_width = line_stroke_width,
                )
                for end_points in zip(
                    line_base_coordinates,
                    line_upper_coordinates,
                )
            ])
        else:
            line_group = VGroup(*[ModifiedLine2(self._set_line_colour, width = line_width) for _ in range(self.num_data)])
            for line, start, end in zip(
                line_group,
                line_base_coordinates,
                line_upper_coordinates,
            ):
                line.put_start_and_end_on(start, end)

        for line in line_group:
            line.to_change_color = True

        return line_group

    def _update_line_colors(self, update_all_line_colors = False):
        for line in self.lines:
            if update_all_line_colors or line.to_change_color:
                self._set_line_colour(line)

    def update_all_line_colors(self):
        self._update_line_colors(True)

    def _change_line_colors(self, *line_indices):
        line_colors = it.cycle(self.selected_line_color)

        for i in line_indices:
            line = self.lines[i]

            line.set_color(
                next(
                    line_colors
                )
            )

            line.to_change_color = True

    def _set_partial_color(self, start, end):
        if start <= end:
            partial_colors = list(reversed(self.partial_colors))
        else:
            partial_colors = self.partial_colors

        start, end = sorted([start, end])

        for i in range(start, end + 1):
            line = self.lines[i]

            line.set_color(
                interpolate_color(
                    partial_colors,
                    i,
                    start,
                    end,
                )
            )

            line.to_change_color = True

    def _update_lines(self):
        for i in np.unique(self.update_line_indices):
            line = self.lines[i]
            start_coordinate = line.find_line_start()
            end_coordinate = start_coordinate + coord(0, self._calculate_line_height(self.data_list[i]))

            line.put_start_and_end_on(
                start_coordinate,
                end_coordinate,
            )

    def update_figure(self, selected_rect_indices, partial_color_range):
        self._update_lines()
        self._update_line_colors()

        if len(partial_color_range) == 2:
            self._set_partial_color(*partial_color_range)

        self._change_line_colors(*selected_rect_indices)
        self.update_line_indices = []

    def scale(self, scale_factor, *args, **kwargs):
        constants_to_be_changed = [
            self.gap_between_lines,
            self.data_width,
            self.data_height,
        ]

        for constant in constants_to_be_changed:
            constant *= scale_factor

        super().scale(scale_factor, *args, **kwargs)

class Sort(Animation):

    _CONFIG = {
        "rate_func": linear,
        "num_iter_per_sec": 500,
        "run_time": 15,
        "time_calculation_method": "num_iter_per_sec",
        "ignore_wrong_code": False,
        "only_test_code": False,
        "supporting_elements": "integers or floats",
        "supporting_objects": [Data],
        "sound_class": AudioFileGenerator,
        "play_sound": True,
        "starting_frequency": 120,
        "ending_frequency": 1212,
        "sound_gain": -5,
        "sound_class_kwargs": {},
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        digest_config(self, kwargs)

        self._check_input_type(data)
        self._initialize_data(data)

        if self.only_test_code:
            self._do_while_testing_code()

        else:
            self._define_important_constants()
            self._generate_swap_list_and_check_for_perfect_code()

            if self.time_calculation_method == "num_iter_per_sec":
                self.run_time = 2 * len(self.list_of_possible_actions) / self.num_iter_per_sec
            else:
                self.num_iter_per_sec = 2 * len(self.list_of_possible_actions) / self.run_time

            if hasattr(self, "data_kwargs"):
                self.data.__dict__.update(self.data_kwargs)

        # if not self.name:
        #     self.name = f"{self.__class__.__name__}({str(self.mobject)})"

    def checking_condition(self, item):
        try:
            float(item)
            return True
        except:
            return False

    def _check_input_type(self, data):
        message = "Class {} works for {} only.".format(self.__class__.__name__, self.supporting_elements)

        if isinstance(data, Data):
            data_list = data.data_list
        else:
            data_list = data

        if type(data_list) in (list, tuple, np.ndarray):
            for item in data_list:
                if not self.checking_condition(item):
                    raise Exception(message)

    def _initialize_data(self, data):
        expected_object_check = any([isinstance(data, cls) for cls in self.supporting_objects])
        if expected_object_check:
            if self.only_test_code:
                self.data = data.data_list
            else:
                self.data = data
        elif type(data) in (list, tuple, np.ndarray):
            self.data = deepcopy(list(data))
            self.only_test_code = True
        else:
            raise Exception(
                "Expected object of type Data, list or ndarray but got {} object".format(
                    type(data)
                )
            )

    def _do_while_testing_code(self):
        initial_data_list = self.data
        data_list = deepcopy(initial_data_list)
        self.num_data = len(data_list)
        self.modified_sort(data_list)

        self._is_sorted = (data_list == sorted(initial_data_list))

        if not self.is_sorted():
            self._print_error_code_message()

    def modified_sort(self, data_list):
        self.sort_if_implemented(data_list)

    def sort_if_implemented(self, data_list):
        if hasattr(self, "sort"):
            self.sort(data_list)
        else:
            raise Exception(
                "sort method is not implemented in class {}.".format(
                    self.__class__.__name__
                )
            )

    def _print_error_code_message(self):
        print("Wrong code written in {} algorithm. Data is not sorted".format(self.__class__.__name__))

    def _generate_swap_list_and_check_for_perfect_code(self):
        self.data_list_to_be_sorted = deepcopy(self.data.data_list)
        self.modified_sort(self.data_list_to_be_sorted)

        if self.data_list_to_be_sorted != self.sorted_data_list:
            self._print_error_code_message()
            if not self.ignore_wrong_code:
                quit()

    def _define_important_constants(self):
        self.mobject = self.data
        self.sorted_data_list = sorted(self.data.data_list)
        self.num_data = self.data.num_data
        self._is_sorted = False
        self.num_steps = 0
        self.generate_lists_of_possible_actions()

    def generate_lists_of_possible_actions(self):
        self.list_of_possible_actions = []
        # self.swap_indices_list = []
        self.selected_object_indices = []
        # self.func_and_parameters_list = []
        self.partial_color_range = []
        self.default_func_list = [do_nothing, [], {}]
        self.value_of_selected_indices_for_adding_sound = []

    def set_value(self, index, value):
        self.data.data_list[index] = value
        self.data.update_line_indices.append(index)
        self.data.num_values_changed += 1

    def clear_value(self, index):
        self.set_value(index, 0)

    def _swap(self, index1, index2):
        i, j = self.data.data_list[index1], self.data.data_list[index2]
        self.set_value(index1, j)
        self.set_value(index2, i)

    def update_action_list(
        self,
        index1 = None,
        index2 = None,
        selected_indices = [],
        func_and_parameters = [],
        partial_color_range = [],
        add_sound_indices = [],
        num_comparisons_done = 0,
        num_values_changed = 0,
    ):
        """
        Write in the following form:

        index1 -> index1:
        Compulsary
        action -> Swap index1 amd index2 in self.data.data_list
        To ignore this step set default value

        index2 -> index2:
        Compulsary
        action -> Swap index1 amd index2 in self.data.data_list
        To ignore this step set default value

        selected_indices -> [indices]:
        Not compulsary
        action -> colours the rectangles of the given indices with the colors self.data.selected_rectangle_color
        To ignore this step set default value

        func_and_parameters -> [func, [arguments], {keyword arguments}]:
        Not compulsary
        action -> applies this function in each step
        You can ignore writing {kwargs}
        To ignore this step set default value

        partial_color_range -> [start_index, end_index]:
        Not compulsary
        action -> colours the rectangles within the given indices with the colors self.data.partial_colors
        To ignore this step set default value

        """

        if not self.only_test_code:
            args = [
                index1,
                index2,
                selected_indices,
                func_and_parameters,
                partial_color_range,
                num_comparisons_done,
                num_values_changed,
            ]

            self.list_of_possible_actions.append(args)

            if self.play_sound:
                self.value_of_selected_indices_for_adding_sound.append(
                    [
                        self.data_list_to_be_sorted[i]
                        for i in add_sound_indices
                    ]
                )

    def _do_on_nth_step(self, n):
        [
            index1,
            index2,
            self.selected_object_indices,
            func_and_parameters,
            self.partial_color_range,
            num_comparisons_done,
            num_values_changed,

        ] = self.list_of_possible_actions[n // 2]

        arbitary_list = self.default_func_list[len(func_and_parameters):]
        func, args, kwargs = list(func_and_parameters) + arbitary_list
        func(*args, **kwargs)

        if None not in (index1, index2):
            self._swap(index1, index2)

        self.data.num_comparisons_done += num_comparisons_done
        self.data.num_values_changed += num_values_changed

    def _nth_step(self, n):
        if n % 2 == 0:
            self.selected_object_indices = self.list_of_possible_actions[n // 2][2]
        else:
            self._do_on_nth_step(n)

    def _next_step(self):
        self._nth_step(self.num_steps)
        self.num_steps += 1

    def next_step(self, update_steps = True):
        n = self.num_steps
        self._next_step()
        self._next_step()
        self.update_figure()
        if not update_steps:
            self.num_steps = n

    def forward_once(self):
        data_list = deepcopy(self.data.data_list)

        while self.data.data_list == data_list and self.sorted_data_list != self.data.data_list:
            self.next_step()

    def is_sorted(self):
        return self._is_sorted

    def update_figure(self):
        self.data.update_figure(
            self.selected_object_indices,
            self.partial_color_range,
        )

    def interpolate_mobject(self, alpha):
        n = 2 * len(self.list_of_possible_actions) * alpha
        next_step_func_called = False

        while self.num_steps < n:
            self._next_step()
            next_step_func_called = True

        if next_step_func_called:
            self.update_figure()

    def begin(self):
        if self.play_sound:
            self.add_sound()
        super().begin()

    def finish(self):
        super().finish()
        self.selected_object_indices = []
        self.partial_color_range = []
        self._is_sorted = True
        self.update_figure()
        self.data.update_all_line_colors()

    def _get_sound_frequencies(self, starting_frequency, ending_frequency):
        frequencies = []

        for item in self.value_of_selected_indices_for_adding_sound:

            frequency_list = [
                match_interpolate(
                    starting_frequency,
                    ending_frequency,
                    self.data.initial_min,
                    self.data.initial_max,
                    value,
                )
                for value in item
            ]

            frequencies.append(frequency_list)

        return frequencies

    def add_sound(self):
        starting_frequency = self.starting_frequency
        ending_frequency = self.ending_frequency

        frequencies = self._get_sound_frequencies(starting_frequency, ending_frequency)

        sound_file_generator = self.sound_class(
            frequencies,
            self.run_time,
            config.leave_progress_bars,
            starting_frequency = self.starting_frequency,
            ending_frequency = self.ending_frequency,
            **self.sound_class_kwargs,
        )

        file_name = sound_file_generator.get_file_name()
        sound_file_generator.generate_file()
        self.scene.add_sound(file_name, gain = self.sound_gain)

    def get_num_iter_per_sec(self):
        return self.num_iter_per_sec / 2

class IntegerSort(Sort):

    _CONFIG = {
        "supporting_elements": "integers",
    }

    def checking_condition(self, item):
        condition1 = super().checking_condition(item)
        if condition1:
            return (item % 1 == 0)
        return False

    def modified_sort(self, data_list):
        self.get_min_and_max_values(data_list)
        if self._min_value < 0:
            self.update_action_list(num_comparisons_done = 1)
            self.shift_values_for_sorting(data_list)

        self.sort_if_implemented(data_list)

        if self._min_value < 0:
            self.update_action_list(num_comparisons_done = 1)
            self.restore_shifted_values(data_list)

    def get_min_and_max_values(self, data_list):
        self._min_value = data_list[0]
        self._max_value = data_list[0]

        for i in range(1, self.num_data):
            self._min_value = min(self._min_value, data_list[i])

            self.update_action_list(
                selected_indices = [i],
                add_sound_indices = [i],
                num_comparisons_done = 1,
            )
            self._max_value = max(self._max_value, data_list[i])

            self.update_action_list(
                selected_indices = [i],
                add_sound_indices = [i],
                num_comparisons_done = 1,
            )

    def shift_values_for_sorting(self, data_list):
        for i in range(self.num_data):
            val = int(data_list[i]) - self._min_value
            data_list[i] = val

            self.update_action_list(
                selected_indices = [i],
                func_and_parameters = [self.set_value, [i, val]],
                add_sound_indices = [i],
                num_values_changed = 1,
            )

    def restore_shifted_values(self, data_list):
        for i in range(self.num_data):
            val = int(data_list[i]) + self._min_value
            data_list[i] = val

            self.update_action_list(
                selected_indices = [i],
                func_and_parameters = [self.set_value, [i, val]],
                add_sound_indices = [i],
                num_values_changed = 1,
            )

    def get_min_value(self):
        return self._min_value

    def get_max_value(self):
        return self._max_value

@sorting_algorithm
class BubbleSort(Sort):

    def sort(self, data_list):
        for i in range(self.num_data - 1):
            flag = 0
            for j in range(self.num_data - i - 1):
                if data_list[j] > data_list[j+1]:
                    swap_list = (j, j+1)
                    swap(data_list, *swap_list)
                    flag = 1
                else:
                    swap_list = (None, None)

                self.update_action_list(
                    *swap_list,
                    selected_indices = [j, j+1],
                    add_sound_indices = [j, j+1],
                    num_comparisons_done = 1,
                )

            if flag == 0:
                break

@sorting_algorithm
class CocktailSort(Sort):

    def sort(self, data_list):
        swapped = True
        start = 0
        end = self.num_data - 1

        while swapped:
            swapped = False

            for i in range(start, end):
                if (data_list[i] > data_list[i+1]) :
                    swap_list = [i, i+1]
                    swap(data_list, *swap_list)

                    self.update_action_list(
                        *swap_list,
                        selected_indices = swap_list,
                        add_sound_indices = swap_list,
                        num_comparisons_done = 1,
                    )

                    swapped=True

            if not swapped:
                break

            swapped = False
            end -= 1

            for i in range(end - 1, start - 1, -1):
                if (data_list[i] > data_list[i+1]):
                    swap_list = [i, i+1]
                    swap(data_list, *swap_list)

                    self.update_action_list(
                        *swap_list,
                        selected_indices = swap_list,
                        add_sound_indices = swap_list,
                        num_comparisons_done = 1,
                    )

                    swapped = True

            start += 1

@sorting_algorithm
class InsertionSort(Sort):

    def sort(self, data_list):
        for i in range(1, self.num_data):
            temp = data_list[i]
            data_list[i] = 0

            self.update_action_list(
                selected_indices = [i, i-1],
                func_and_parameters = (self.clear_value, [i]),
                add_sound_indices = [i, i-1],
            )

            j = i - 1

            while j >= 0 and data_list[j] > temp:
                data_list[j+1] = data_list[j]

                self.update_action_list(
                    selected_indices = [j],
                    func_and_parameters = (self.set_value, [j+1, data_list[j]]),
                    add_sound_indices = [j],
                    num_comparisons_done = 2,
                )

                j -= 1
            data_list[j+1] = temp

            self.update_action_list(
                selected_indices = [j+1],
                func_and_parameters = (self.set_value, [j+1, temp]),
                add_sound_indices = [j+1],
            )


@sorting_algorithm
class SelectionSort(Sort):

    _CONFIG = {
        "data_kwargs": {
            "selected_rectangle_color": [YELLOW, WHITE],
        },
    }

    def sort(self, data_list):
        for i in range(self.num_data - 1):
            min_index = i
            for j in range(i + 1, self.num_data):
                if data_list[j] < data_list[min_index]:
                    min_index = j

                self.update_action_list(
                    selected_indices = [min_index, j],
                    add_sound_indices = [i],
                    num_comparisons_done = 1,
                )

            if min_index != i:
                swap_list = (i, min_index)
                swap(data_list, *swap_list)
            else:
                swap_list = (None, None)

            self.update_action_list(
                *swap_list,
                selected_indices = [min_index, i],
                add_sound_indices = [i],
                num_comparisons_done = 1,
            )

@sorting_algorithm
class CycleSort(Sort):

    def sort(self, data_list):
        for i in range(0, self.num_data - 1):
            item = data_list[i]

            position = i
            for j in range(i + 1, self.num_data):
                if data_list[j] < item:
                    position += 1

                self.update_action_list(
                    selected_indices = [i, j],
                    add_sound_indices = [i, j],
                    num_comparisons_done = 1,
                )

            self.update_action_list(
                selected_indices = [i, position],
                add_sound_indices = [position],
                num_comparisons_done = 1,
            )

            if position == i:
                continue

            while item == data_list[position]:

                self.update_action_list(
                    selected_indices = [i, position],
                    add_sound_indices = [position],
                    num_comparisons_done = 1,
                )

                position += 1

            val = item
            data_list[position], item = item, data_list[position]

            self.update_action_list(
                selected_indices = [i],
                func_and_parameters = [self.clear_value, [i]],
                add_sound_indices = [i]
            )

            self.update_action_list(
                selected_indices = [position],
                func_and_parameters = [self.set_value, [position, val]],
                add_sound_indices = [position]
            )

            while position != i:
                self.update_action_list(num_comparisons_done = 1)
                position = i
                for j in range(i + 1, self.num_data):
                    if data_list[j] < item:
                        position += 1

                    self.update_action_list(
                        selected_indices = [i, j, position],
                        add_sound_indices = [j],
                        num_comparisons_done = 1,
                    )

                while item == data_list[position]:

                    self.update_action_list(
                        selected_indices = [i, position],
                        add_sound_indices = [position],
                        num_comparisons_done = 1,
                    )
                    position += 1

                val = item
                data_list[position], item = item, data_list[position]

                self.update_action_list(
                    selected_indices = [i],
                    func_and_parameters = [self.clear_value, [i]],
                    add_sound_indices = [i],
                )

                self.update_action_list(
                    selected_indices = [position],
                    func_and_parameters = [self.set_value, [position, val]],
                    add_sound_indices = [position],
                )

@sorting_algorithm
class QuickSort(Sort):

    _CONFIG = {
        "num_iter_per_sec": 200,

        "data_kwargs": {
            "selected_rectangle_color": [YELLOW, WHITE, WHITE],
        },
    }

    def sort(self, data_list):
        self.partial_sort(data_list, 0, self.num_data - 1)

    def partial_sort(self, data_list, lb, ub):

        random_indices = [
            np.clip(lb, 0, self.num_data - 1),
            np.clip(ub, 0, self.num_data - 1)
        ] if lb < ub else []

        self.update_action_list(
            partial_color_range = random_indices,
            add_sound_indices = random_indices,
            num_comparisons_done = 1,
        )

        if lb < ub:
            loc = self.partition(data_list, lb, ub)
            self.partial_sort(data_list, lb, loc - 1)
            self.partial_sort(data_list, loc + 1, ub)

    def partition(self, data_list, lb, ub):
        pivot = data_list[lb]
        start = lb
        end = ub

        while start < end:

            self.update_action_list(
                selected_indices = [lb, start, end],
                partial_color_range = [lb, ub],
                add_sound_indices = [start, end],
                num_comparisons_done = 1,
            )


            while data_list[start] <= pivot:

                self.update_action_list(
                    selected_indices = [lb, start],
                    partial_color_range = [lb, ub],
                    add_sound_indices = [start],
                    num_comparisons_done = 1,
                )

                start += 1

                self.update_action_list(
                    partial_color_range = [lb, ub],
                    add_sound_indices = [start if start <= ub else ub, ub],
                    num_comparisons_done = 1,
                )

                if start > ub:
                    start = ub
                    break

            while data_list[end] > pivot:

                self.update_action_list(
                    selected_indices = [lb, end],
                    partial_color_range = [lb, ub],
                    add_sound_indices = [end],
                    num_comparisons_done = 1,
                )

                end -= 1

                self.update_action_list(
                    partial_color_range = [lb, ub],
                    add_sound_indices = [end if end >= lb else lb, lb],
                    num_comparisons_done = 1,
                )

                if end < lb:
                    end = lb
                    break

            if start < end:
                swap_list = (start, end)
                swap(data_list, *swap_list)
            else:
                swap_list = (None, None)

            self.update_action_list(
                *swap_list,
                selected_indices = [lb, start, end],
                partial_color_range = [lb, ub],
                add_sound_indices = [start, end],
                num_comparisons_done = 1,
            )

        swap_list = (lb, end)
        swap(data_list, *swap_list)

        self.update_action_list(
            *swap_list,
            selected_indices = [lb, end], 
            partial_color_range = [lb, ub],
            add_sound_indices = [end],
        )


        return end

@sorting_algorithm
class MergeSort(Sort):

    _CONFIG = {
     "num_iter_per_sec": 200,
    }

    def sort(self, data_list):
        self.partial_sort(data_list, 0, self.num_data - 1)

    def partial_sort(self, data_list, lb, ub):
        self.update_action_list(num_comparisons_done = 1)
        if lb < ub:
            mid = (lb + ub) // 2
            self.partial_sort(data_list, lb, mid)
            self.partial_sort(data_list, mid + 1, ub)
            self.merge(data_list, lb, ub)

    def merge(self, data_list, lb, ub):
        mid = (lb + ub) // 2
        i, j, k, data_list_copy = lb, mid + 1, lb, deepcopy(data_list)

        while i <= mid and j <= ub:

            if data_list[i] <= data_list[j]:
                data_list_copy[k] = data_list[i]
                i += 1
            else:
                data_list_copy[k] = data_list[j]
                j += 1

            self.update_action_list(
                selected_indices = [i - 1, j - 1, mid, ub],
                partial_color_range = [lb, ub],
                add_sound_indices = [i - 1, j - 1],
                num_comparisons_done = 3,
            )

            k += 1

        if i > mid:

            if i < self.num_data:
                selected_indices = [i]
            else:
                selected_indices = [self.num_data - 1]

            self.update_action_list(
                selected_indices = selected_indices,
                partial_color_range = [lb, ub],
                num_comparisons_done = 2,
            )


            while j <= ub:
                data_list_copy[k] = data_list[j]

                self.update_action_list(
                    selected_indices = [j, ub],
                    partial_color_range = [lb, ub],
                    num_comparisons_done = 1,
                )

                j += 1
                k += 1
        else:
            while i <= mid:
                data_list_copy[k] = data_list[i]

                self.update_action_list(
                    selected_indices = [i, mid],
                    partial_color_range = [lb, ub],
                    num_comparisons_done = 1,
                )

                i += 1
                k += 1

        for i in range(lb, ub + 1):
            val = data_list_copy[i]
            data_list[i] = val
            self.update_action_list(
                selected_indices = [i],
                func_and_parameters = [self.set_value, [i, val]],
                partial_color_range = [lb, ub],
                add_sound_indices = [i]
            )


@sorting_algorithm
class HeapSort(Sort):

    _CONFIG = {
        "num_iter_per_sec": 200,
    }

    def sort(self, data_list):
        for i in range(self.num_data // 2 - 1, -1, -1):
            self.heapify(data_list, self.num_data, i)

        for i in range(self.num_data - 1, 0, -1):
            swap_list = (i, 0)
            swap(data_list, *swap_list)

            self.update_action_list(
                *swap_list,
                selected_indices = [i, 0],
                add_sound_indices = [i],
            )

            self.heapify(data_list, i, 0)

    def heapify(self, data_list, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        for item in (l, r):
            if item < n:
                if data_list[largest] < data_list[item]:
                    largest = item

            self.update_action_list(
                selected_indices = [largest, np.clip(item, 0, self.num_data - 1)],
                add_sound_indices = [largest],
                num_comparisons_done = 2,
            )

        if largest != i:
            swap_list = (i, largest)
            swap(data_list, *swap_list)

            self.update_action_list(
                *swap_list,
                selected_indices =[i, largest],
                add_sound_indices = [largest],
                num_comparisons_done = 1,
            )

            self.heapify(data_list, n, largest)

        else:
            self.update_action_list(
                selected_indices =[i, largest],
                add_sound_indices = [largest],
                num_comparisons_done = 1,
            )

@sorting_algorithm
class ShellSort(Sort):

    _CONFIG = {
        "num_iter_per_sec": 350,
    }

    def get_initial_gap_and_gap_func(self):
        gap = self.num_data // 2

        def gap_func(gap):
            return gap // 2

        return gap, gap_func

    def sort(self, data_list):
        gap, gap_func = self.get_initial_gap_and_gap_func()

        while gap >= 1:
            self.update_action_list(num_comparisons_done = 1)

            for j in range(gap, self.num_data):
                i = j - gap

                while i >= 0:

                    self.update_action_list(
                        selected_indices = [i],
                        add_sound_indices = [i],
                        num_comparisons_done = 1,
                    )

                    if data_list[i+gap] > data_list[i]:
                        swap_list = (None, None)
                        break
                    else:
                        swap_list = (i+gap, i)
                        swap(data_list, *swap_list)

                    self.update_action_list(
                        *swap_list,
                        selected_indices = [i+gap, i],
                        add_sound_indices = [i+gap, i],
                        num_comparisons_done = 1,
                    )

                    i -= gap

            gap = int(
                round(
                    gap_func(gap)
                )
            )

@sorting_algorithm
class StoogeSort(Sort):
    pass

@sorting_algorithm
class BitonicSort(Sort):
    pass

@sorting_algorithm
class CountingSort(IntegerSort):

    _CONFIG = {
        "num_iter_per_sec": 250,
    }

    def sort(self, data_list):
        cumulative_count_dict = self.get_cumulative_count_dict(data_list)
        data_list_copy = deepcopy(data_list)

        for i in range(self.num_data - 1, -1, -1):
            val = data_list[i]
            cumulative_count_dict[val] -= 1
            index = cumulative_count_dict[val]
            data_list_copy[index] = val

            self.update_action_list(
                selected_indices = [i],
                add_sound_indices = [i],
            )

        for i in range(self.num_data):
            val = data_list_copy[i]
            data_list[i] = val

            self.update_action_list(
                selected_indices = [i],
                func_and_parameters = [self.set_value, [i, val]],
                add_sound_indices = [i],
            )

    def get_count_dict(self, data_list):
        count_dict = {}

        def increment_value(index):
            nonlocal data_list, count_dict
            val = data_list[index]

            if val not in count_dict:
                count_dict[val] = 0

            count_dict[val] += 1

        for i in range(self.num_data):
            increment_value(i)
            self.update_action_list(
                selected_indices = [i],
                add_sound_indices = [i],
            )

        return count_dict

    def get_cumulative_count_dict(self, data_list):
        count_dict = self.get_count_dict(data_list)
        count__dict__to_sorted_list = sorted(list(count_dict))
        cumulative_count_dict = deepcopy(count_dict)

        for i in range(1, len(count__dict__to_sorted_list)):
            j = count__dict__to_sorted_list[i]
            prev_j = count__dict__to_sorted_list[i-1]

            cumulative_count_dict[j] += cumulative_count_dict[prev_j]
            self.update_action_list()

        return cumulative_count_dict

@sorting_algorithm
class RadixSort(IntegerSort):

    _CONFIG = {
        "num_iter_per_sec": 200,
    }

    def sort(self, data_list):
        max_value = self.get_max_value()
        position = 1

        while max_value // position > 0:
            self.partial_sort(data_list, position)
            position *= 10

    def partial_sort(self, data_list, position):
        data_list_copy = deepcopy(data_list)
        count_array = [0] * 10

        for i in range(self.num_data):
            count_array[(data_list[i] // position) % 10] += 1

            self.update_action_list(
                selected_indices = [i],
                add_sound_indices = [i],
            )

        for i in range(1, 10):
            count_array[i] += count_array[i-1]
            self.update_action_list()

        for i in range(self.num_data - 1, -1, -1):
            count_array[(data_list[i]//position) % 10] -= 1
            val = count_array[(data_list[i]//position) % 10]
            data_list_copy[val] = data_list[i]

            self.update_action_list(
                selected_indices = [i],
                add_sound_indices = [i],
            )

        for i in range(self.num_data):
            val = data_list_copy[i]
            data_list[i] = val

            self.update_action_list(
                selected_indices = [i],
                func_and_parameters = [self.set_value, [i, val]],
                add_sound_indices = [i],
            )

class ShowIsSorted(Animation):

    _CONFIG = {
        "rate_func": linear,
        "run_time": 2,
        "partial_colors": ["#00AA00", "#005000"],
        # "partial_colors": [
        #   "#AF0850",
        #   "#A5095A",
        #   "#9B0A64",
        #   "#910B6E",
        #   "#870C78",
        #   "#7D0D82",
        #   "#730E8C",
        #   "#690F96",
        # ],
        "sound_class": AudioFileGenerator,
        "play_sound": True,
        "starting_frequency": 120,
        "ending_frequency": 1212,
        "sound_gain": -5,
        "sound_class_kwargs": {},
    }

    def __init__(self, data, **kwargs):
        self.run_time = None
        super().__init__(data, **kwargs)
        digest_config(self, kwargs)

        self._check_input_type(data)
        self.data = data
        self.mobject = self.data
        self.sorted_data_list = sorted(self.data.data_list)
        self.num_data = self.data.num_data

        if not self.run_time:
            self.run_time = self._calculate_run_time()

        self._check_if_sorted()

    def _check_input_type(self, data):
        message = "Expected Data object but got {} object".format(type(data))
        if not isinstance(data, Data):
            raise Exception(message)

    def _check_if_sorted(self):
        if self.data.data_list != self.sorted_data_list:
            raise Exception("Data is not sorted.")

    def _calculate_run_time(self):
        return np.clip(self.num_data / 100, 0.4, 1.5)

    def interpolate_mobject(self, alpha):
        n = int(round(
            (self.num_data - 1) * alpha
        ))

        self.data.update_figure([n], [0, n])

    def begin(self):
        if self.play_sound:
            self.add_sound()

        self.data.partial_colors, self.prev_partial_colors = self.partial_colors, self.data.partial_colors

        super().begin()

    def finish(self):
        super().finish()
        self.data.partial_colors = self.prev_partial_colors
        self.data.update_all_line_colors()

    def _get_sound_frequencies(self, starting_frequency, ending_frequency):
        frequencies = match_interpolate(
            starting_frequency,
            ending_frequency,
            min(self.data.data_list),
            max(self.data.data_list),
            np.array(self.data.data_list),
        )

        return [[i] for i in frequencies]

    def add_sound(self):
        starting_frequency = self.starting_frequency
        ending_frequency = self.ending_frequency

        frequencies = self._get_sound_frequencies(starting_frequency, ending_frequency)

        sound_file_generator = self.sound_class(
            frequencies,
            self.run_time,
            config.leave_progress_bars,
            starting_frequency = self.starting_frequency,
            ending_frequency = self.ending_frequency,
            **self.sound_class_kwargs,
        )

        file_name = sound_file_generator.get_file_name()
        sound_file_generator.generate_file()
        self.scene.add_sound(file_name, gain = self.sound_gain)

class SortingScene(Scene):

    _CONFIG = {
        "text_scale_factor": 3,
        "sort_kwargs": {},
        "is_sorted_kwargs": {},
        "buff_in_labels": 0.4,
        "buff_in_complete_label": 0.1,
        # "decimal_number_scale_facror": 0.6,
        "decimal_number_scale_facror": 1,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        digest_config(self, kwargs)

    def play(self, *animations, **kwargs):
        for animation in animations:
            if any([isinstance(animation, cls) for cls in (Sort, ShowIsSorted)]):
                animation.scene = self

        super().play(*animations, **kwargs)

    def create_random_data(self, num_data, seed = None):
        if seed != None:
            np.random.seed(seed)

        data_list = np.linspace(0, num_data - 1, num_data, dtype = int)
        np.random.shuffle(data_list)

        return data_list

    def generate_scene_text(self):
        self.text_objects = {}

        for cls in SORTING_ALGORITHMS:
            name = ""

            for letter in cls.__name__:
                if letter.isupper():
                    name += " "
                name += letter

            name = name.strip()

            self.text_objects[cls] = Tex(name).scale(self.text_scale_factor)

    def play_sort_animation(self, sort, data):
        self.play(sort(data, **self.sort_kwargs))
        self.play(ShowIsSorted(data, **self.is_sorted_kwargs))

    def get_num_comparisions_label(self, data):
        text = VGroup(
            Tex("Number of comparisons done:"),
            DecimalNumber(0, num_decimal_places = 0).scale(self.decimal_number_scale_facror),
        ).arrange(RIGHT, buff = self.buff_in_labels)

        text.add_updater(
            lambda mob: mob[-1].set_value(data.num_comparisons_done)
        )

        return text

    def get_num_values_changed_label(self, data):
        text = VGroup(
            Tex("Number of times array modified:"),
            DecimalNumber(0, num_decimal_places = 0).scale(self.decimal_number_scale_facror),
        ).arrange(RIGHT, buff = self.buff_in_labels)

        text.add_updater(
            lambda mob: mob[-1].set_value(data.num_values_changed)
        )

        return text

    def get_complete_label(self, data):
        complete_label = VGroup(
            self.get_num_comparisions_label(data),
            self.get_num_values_changed_label(data),
        )

        complete_label.arrange(
            DOWN,
            buff = self.buff_in_complete_label,
            aligned_edge = LEFT,
        )

        return complete_label