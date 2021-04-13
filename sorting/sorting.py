import os, sys
sys.path.append(os.path.split(__file__)[0])
from classes_and_functions import *
sys.setrecursionlimit(10**5)

class TestSingleScene(SortingScene):

    _CONFIG = {
        "num_data": 1000,
        "sort": CycleSort,
        "seed": None,
        "data_class": Data,
        # "sort_data": False,

        "data_kwargs":{
            "data_height": 6,
        },

        "sort_kwargs": {
            "time_calculation_method": "run_time",
            "num_iter_per_sec": 5,
            "run_time": 15,
            "play_sound": True,
            # "sound_class": AudioFileGeneratorTrial,
        },
    }

    def construct(self):
        if not hasattr(self, "sort_kwargs"):
            self.sort_kwargs = {}
        if not hasattr(self, "seed"):
            self.seed = None
        if not hasattr(self, "sort_data"):
            self.sort_data = True

        data_list = np.linspace(0, self.num_data - 1, self.num_data, dtype = int)
        np.random.shuffle(data_list)

        self.data_group = self.data_class(
            data_list,
            **self.data_kwargs
        )
        self.data_group.to_edge(DOWN, buff = 0.2)

        label = self.get_complete_label(self.data_group)
        label.to_corner(UL)

        self.add(self.data_group, label)
        self.play_scenes()

    def play_scenes(self):
        self.wait(2)

        if self.sort_data:
            self.play_sort_animation(self.sort, self.data_group)

        self.wait(5)

class TestForPerfectCode(SortingScene):
    _CONFIG = {
        "num_data": 10,
        "ignore_error": False,
        "text_scale_factor": 3,
        "data_class": Data,

        "sort_kwargs": {
            "time_calculation_method": "run_time",
            "run_time": 1,
			"play_sound": False,
        },
    }

    def construct(self):
        if not hasattr(self, "sort_kwargs"):
            self.sort_kwargs = {}

        self.data_list = self.create_random_data(self.num_data)

        self.generate_scene_text()
        for sort in SORTING_ALGORITHMS:
            self.play_scenes(sort)

    def create_random_data(self, num_data, seed = None):
        if seed != None:
            np.random.seed(seed)

        data_list = [np.random.randint(0, 9999) for _ in range(self.num_data)]

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

    def play_scenes(self, sort):
        data = self.data_class(
            deepcopy(self.data_list),
            total_width = 10,
        )

        text = self.text_objects[sort]
        self.add(text)
        self.wait()
        self.remove(text)

        self.add(data)

        def play_func():
            self.play_sort_animation(sort, data)

        if self.ignore_error:
            try:
                play_func()

            except Exception as e:
                print(e)
        else:
            play_func()

        self.wait()

        self.remove(data)

class PlayAllSortAlgorithms(SortingScene):
    _CONFIG = {
        "num_data": 1000,
        "text_scale_factor": 3,
        "play_short_trial": False,
        "anim_run_time": 2,
        "data_class": Data,
        "sorting_algorithms": SORTING_ALGORITHMS,
        # "sorting_algorithms": [QuickSort],

        "data_kwargs": {
            "data_height": 6.3,
            # "data_height": 5,
        },

        "sort_kwargs": {
            "time_calculation_method": "run_time",
            "run_time": 25,
            "sound_class": AudioFileGenerator,
            # "play_sound": False,
        },
    }

    def construct(self):

        # self.add(SurroundingRectangle(self.camera.frame, buff = 0.5))

        if self.play_short_trial:
            self.sort_kwargs.update(
                dict(
                    time_calculation_method = "run_time",
                )
            )
            self.anim_run_time = 1

        self.data_list = self.create_random_data(self.num_data)

        self.generate_scene_text()
        for sort in self.sorting_algorithms:
            self.play_scenes(sort)

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

    def play_scenes(self, sort):
        data = self.data_class(
            deepcopy(self.data_list),
            **self.data_kwargs,
        )
        data.to_edge(DOWN, buff = 0.1)

        label = self.get_complete_label(data)
        label.to_corner(UL)

        text = self.text_objects[sort]

        self.play(Write(text, run_time = self.anim_run_time))
        self.wait()
        self.play(FadeOut(text, shift_vect = DOWN, run_time = self.anim_run_time))
        self.wait()

        self.play(AnimationGroup(
            Create(data),
            Write(label),
            run_time = self.anim_run_time,
            lag_ratio = 0,
        ))

        self.wait()
        self.play_sort_animation(sort, data)
        self.wait(3)

        self.play(AnimationGroup(
            FadeOut(data),
            FadeOut(label, shift_vect = DOWN),
            run_time = self.anim_run_time,
            lag_ratio = 0,
            ))

        self.wait()

        del data, label

class Debugging(SortingScene):
    _CONFIG = {
        "num_data": 10,
        "ignore_error": False,
        "text_scale_factor": 3,
        "data_class": Data,

        "sort_kwargs": {
            "time_calculation_method": "run_time",
            "run_time": 1,
			"play_sound": False,
        },
    }

    def construct(self):
        if not hasattr(self, "sort_kwargs"):
            self.sort_kwargs = {}

        self.data_list = self.create_random_data(self.num_data)

        self.generate_scene_text()
        for sort in SORTING_ALGORITHMS:
            self.play_scenes(sort)

    def create_random_data(self, num_data, seed = None):
        if seed != None:
            np.random.seed(seed)

        data_list = [np.random.randint(0, 9999) for _ in range(self.num_data)]

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

    def play_scenes(self, sort):
        data = self.data_class(
            deepcopy(self.data_list),
            total_width = 10,
        )

        text = self.text_objects[sort]
        self.add(text)
        self.wait()
        self.remove(text)

        self.add(data)

        # def play_func():
        #     self.play_sort_animation(sort, data)

        # if self.ignore_error:
        #     try:
        #         play_func()

        #     except Exception as e:
        #         print(e)
        # else:
        #     play_func()

        self.play(sort(data))

        self.wait()

        self.remove(data)

class Debugging2(SortingScene):

    def construct(self):
        data = Data(self.create_random_data(10))
        self.add(data)
        self.play(MergeSort(data, run_time = 15, time_calculation_method = "run_time"))

# TestForPerfectCode().render()