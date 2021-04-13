from manim import *

def digest_config(self, kwargs):
    for cls in reversed(self.__class__.mro()):
        if hasattr(cls, "CONFIG"):
            if self.CONFIG != {}:
                self.__dict__.update(cls.CONFIG)
    if kwargs != {}:
        self.__dict__.update(kwargs)

class DerivativeScene1(GraphScene):
    CONFIG={
        "graph_color": None,
        "graph_range": (None, None),
        "graph_func_name": None,

        "initial_dx": 1,
        "final_dx": 0.01,
        "run_time": 20,
    }

    def construct(self):
        digest_config(self, {})
        self.play_animation(self.initial_dx, self.final_dx, 1)

    def play_animation(self, initial_dx, final_dx, scale_factor):
        self.setup_axes(animate=True)
        self.define_objects(initial_dx)

        self.graph_label.scale(1.5).to_edge(UR, buff = 0.5)
        self.dx_label.scale(scale_factor).to_corner(UL, buff = 0.5)
        self.num_rectangles_label.scale(scale_factor).to_corner(UL, buff = 0.5).shift(0.75*DOWN)

        self.play(
            *[
                Method(item)
                for Method, item in zip(
                    [ShowCreation, Write],
                    [self.graph, self.graph_label],
                )
            ],
            run_time = 2,
        )

        self.play(
            *[
                Method(item)
                for Method, item in zip(
                    [Write, Write, Write],
                    [self.riemann_rect, self.dx_label, self.num_rectangles_label],
                )
            ],
            run_time = 2,
        )

        self.wait()

        self.add_required_updaters()

        self.play(
            ApplyMethod(
                self.riemann_rect_width.set_value,
                final_dx,
            ),
            run_time = self.run_time,
        )

        self.clear_all_updaters()

        self.wait(5)

    def define_objects(self, initial_dx):
        self.graph = self.get_graph(self.graph_func, self.graph_color, *self.graph_range)

        self.riemann_rect_width = ValueTracker().set_value(initial_dx)

        if self.graph_func_name != None:
            self.graph_label = Tex(
                "Graph: ${}$".format(self.graph_func_name)
            )
        else:
            self.graph_label = VMobject()

        self.riemann_rect = self.get_riemann_rectangles(
            self.graph,
            *self.graph_range,
            dx = self.riemann_rect_width.get_value(),
        )

        self.dx_label = VGroup(
            Tex("$dx=$"),
            DecimalNumber(
                self.riemann_rect_width.get_value(),
                num_decimal_places = 3,
            ),
        ).arrange(RIGHT)

        self.num_rectangles_label = VGroup(
            Tex("Number of Rectangles ="),
            DecimalNumber(
                np.ceil(
                    np.diff(self.graph_range)[0]/self.riemann_rect_width.get_value(),
                ),
                num_decimal_places = 0,
            )
        ).arrange(RIGHT)

    def add_required_updaters(self):
        self.riemann_rect.add_updater(
            lambda mob: mob.become(
                self.get_riemann_rectangles(
                    self.graph,
                    *self.graph_range,
					stroke_width = match_interpolate(
						1,
						0,
						self.initial_dx,
						self.final_dx,
						self.riemann_rect_width.get_value(),
					),
                    dx = self.riemann_rect_width.get_value()
                )
            )
        )

        self.dx_label.add_updater(
            lambda mob: mob[-1].set_value(
                self.riemann_rect_width.get_value()
            )
        )

        self.num_rectangles_label.add_updater(
            lambda mob: mob[-1].set_value(
                np.ceil(
                    np.diff(self.graph_range)[0]/self.riemann_rect_width.get_value(),
                )
            )
        )

    def clear_all_updaters(self):
        for mob in [self.riemann_rect, self.dx_label, self.num_rectangles_label]:
            mob.clear_updaters()

class DerivativeScene2(DerivativeScene1):
    CONFIG={
        "x_min": -5,
        "x_max": 5,
        "y_min": -3,
        "y_max": 3,
        "graph_origin": ORIGIN,
    }

class SinGraphScene(DerivativeScene2):
    CONFIG={
        "graph_func": lambda x: 2*np.sin(x),
        "graph_range": (-1.5*PI, 1.5*PI),
        "graph_func_name": "\\sin(x)",
    }

class ExponentialGraphScene(DerivativeScene1):
    CONFIG={
        "x_max": 5,
        "y_max": np.exp(5),
        "y_min": -np.exp(5)/10,
        "y_tick_frequency": np.exp(5)/10,
        "y_axis_height": 5,

        "graph_origin": 3 * DOWN + 4 * LEFT,
        "graph_func": lambda x: np.exp(x),
        "graph_range": (-1, 5),
        "graph_func_name": "e^x",

        "final_dx": 0.03,
    }