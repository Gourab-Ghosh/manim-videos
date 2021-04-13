from manim import *
import os

def digest_config(self, kwargs = {}):
    for cls in reversed(self.__class__.mro()):
        if hasattr(cls, "CFG"):
            if self.CFG != {}:
                self.__dict__.update(cls.CFG)
    if kwargs != {}:
        self.__dict__.update(kwargs)

def turn_animation_into_updater(animation, cycle=False, scene = None, delete_mob = False, **kwargs):
    mobject = animation.mobject
    remover = scene != None and animation.remover
    if remover:
        scene.add(mobject)
    animation.__dict__.update(**kwargs)
    animation.suspend_mobject_updating = False
    animation.begin()
    animation.total_time = 0

    def update(m, dt):
        run_time = animation.get_run_time()
        time_ratio = animation.total_time / run_time
        if cycle:
            alpha = time_ratio % 1
        else:
            alpha = np.clip(time_ratio, 0, 1)
            if alpha >= 1:
                if scene != None:
                    if remover:
                        scene.remove(m)
                animation.finish()
                m.remove_updater(update)
                if delete_mob:
                    del m
                return
        animation.interpolate(alpha)
        animation.update_mobjects(dt)
        animation.total_time += dt

    mobject.add_updater(update)
    return mobject

class Flash(AnimationGroup):
    CFG = {
        "line_length": 0.2,
        "num_lines": 12,
        "flash_radius": 0.3,
        "line_stroke_width": 3,
        "run_time": 1,
        "angle_range" : [0, TAU],
        "time_width" : None,
    }

    def __init__(self, point, color=YELLOW, **kwargs):
        self.point = point
        self.color = color
        digest_config(self, kwargs)

        if self.time_width == None:
            self.time_width = self.run_time
        if "vector" in kwargs:
            if type(self.angle_range) not in (int, float):
                self.angle_range = TAU
            self.set_angle_range_from_vector_and_angle_range(**kwargs)
        else:
            self.angle_range = sorted(list(self.angle_range))

        self.lines = self.create_lines()
        animations = self.create_line_anims()
        super().__init__(*animations, group=self.lines, **kwargs)

    def set_angle_range_from_vector_and_angle_range(self, vector, r_type = "half", **kwargs):
        r_type = r_type.lower().strip()
        if r_type == "full":
            self.angle_range /= 2
        if type(vector) is np.ndarray:
            vector_angle = angle_of_vector(vector)
        else:
            vector_angle = vector.get_angle()
        self.angle_range = [vector_angle - abs(self.angle_range), vector_angle + abs(self.angle_range)]

    def create_lines(self):
        lines = VGroup()
        starting_angle = self.angle_range[0]
        ending_angle = self.angle_range[1]
        angle_range = ending_angle - starting_angle
        for angle in np.linspace(starting_angle, ending_angle, self.num_lines, dtype = "float64"):
            line = Line(ORIGIN, self.line_length * RIGHT)
            line.shift((self.flash_radius - self.line_length) * RIGHT)
            line.rotate(angle, about_point = ORIGIN)
            lines.add(line)
        lines.set_color(self.color)
        lines.set_stroke(width = self.line_stroke_width)
        lines.shift(self.point)
        return lines

    def create_line_anims(self):
        return [
            ShowCreationThenDestruction(line, rate_func = self.rate_func, time_width = self.time_width)
            for line in self.lines
        ]

def vec_len(vec):
    return np.sqrt((vec**2).sum())

def within(Lower_value, Number, Upper_value, Type_of_inequality = "strict"):
    Type_of_inequality = Type_of_inequality.lower().strip()
    if Type_of_inequality == "strict":
        return Number > Lower_value and Number < Upper_value
    else:
        return Number >= Lower_value and Number <= Upper_value

distance = lambda vec1, vec2: vec_len(vec1 - vec2)

def unit_vec(vec, iterations = 100):
    if (vec == np.array([0, 0, 0])).all():
        return vec
    unit_vector = vec / vec_len(vec)
    return unit_vector

def det(matrix):
    width = len(matrix)
    if width == 1:
       return matrix[0][0]
    else:
       sign = -1
       sum = 0
       for i in range(width):
           m = []
           for j in range(1, width):
               buff = []
               for k in range(width):
                   if k != i:
                       buff.append(matrix[j][k])
               m.append(buff)
           sign *= -1
           sum += sign * matrix[0][i] * det(m)
       return sum

def in_same_direction(vec1, vec2, tolerance = 1e-5):
    return vec_len(unit_vec(vec1) - unit_vec(vec2)) <= tolerance

def in_opposite_direction(vec1, vec2, tolerance = 1e-5):
    return vec_len(unit_vec(vec1) + unit_vec(vec2)) <= tolerance

class Particle(Circle):
    CFG = {
    "stroke_width" : 3,
    "velocity" : np.array([0, 0, 0]),
    "is_conductive" : True,
    }
    g = 0
    default_radius = 0.4
    max_fill_opacity = 0.5
    max_stroke_width = 15
    neutrality_tolerance = 0
    group_of_particles = VGroup()
    consider_collision = True
    positive_charge_colour = RED
    positive_charge_sign = "+"
    negative_charge_colour = BLUE
    negative_charge_sign = "-"
    zero_charge_colour = GREEN
    zero_charge_sign = "N"
    attraction_constant = 50
    collision_force_constant = 1

    def __init__(self, scene, mass, charge, initial_position = ORIGIN, **kwargs):
        digest_config(self, kwargs)
        super().__init__(
            stroke_width = self.stroke_width,
            radius = self.default_radius,
            **kwargs,
        )

        self.move_to(initial_position)

        self.scene = scene
        self._mass = mass
        self._charge = charge

        if not "_radius" in kwargs:
            self._radius = self.default_radius

        self.group_of_particles.add(self)

    @classmethod
    def create_objects(cls, list_of_particle_descriptions, setup = True):
        for args in list_of_particle_descriptions:
            args = list(args)
            kwargs = {}
            for description in args:
                if type(description) == dict:
                    kwargs.update(args.pop(args.index(description)))
            cls(*args, **kwargs)
        if setup:
            cls.setup()

    @classmethod
    def setup(cls):
        cls.neutrality_tolerance = abs(cls.neutrality_tolerance)
        cls.initial_mass_list = [particle.mass for particle in cls.group_of_particles]
        cls.initial_charge_magnitude_list = [abs(particle.charge) for particle in cls.group_of_particles]
        for self in cls.group_of_particles:
            self.mass = self._mass
            self.charge = self._charge
            self.add(self.sign)

    @classmethod
    def get_sum_of_charges(cls):
        return sum([particle.charge for particle in cls.group_of_particles])

    @classmethod
    def delete_particles(cls, list_of_particles = None, parmanently_delete_particles = False):
        if list_of_particles == None:
            list_of_particles = list(cls.group_of_particles)
        for particle in list_of_particles:
            cls.group_of_particles.remove(particle)
            if parmanently_delete_particles:
                del particle

    @classmethod
    def calculate_electric_field_on_point(cls, point):
        electric_field = np.array([0, 0, 0], dtype = "float64")
        for particle in cls.group_of_particles:
            k = cls.attraction_constant
            q = particle.charge
            r_vector = point - particle.position
            if vec_len(r_vector) > vec_len(particle._radius):
                electric_field += (k * q / vec_len(r_vector) ** 3) * r_vector
            else:
                electric_field = np.array([0, 0, 0], dtype = "float64")
        if np.isnan(vec_len(electric_field)):
            electric_field = np.array([0, 0, 0], dtype = "float64")
        return electric_field

    @classmethod
    def add_particle_updater(cls, surrounding_rectangle = None):
        if surrounding_rectangle != None:
            rectangle_sides_list = [Line(surrounding_rectangle.point_from_proportion(i / 4), surrounding_rectangle.point_from_proportion(((i + 1) % 4) / 4)) for i in range(4)]
        else:
            rectangle_sides_list = []

        def update_vgroup(vgroup, dt):
            nonlocal rectangle_sides_list
            for particle in vgroup:
                particle.update_position_velocity_and_charge(dt, rectangle_sides_list)

        cls.group_of_particles.add_updater(update_vgroup)

    @property
    def position(self):
        return self.get_center()

    @position.setter
    def position(self, point):
        self.move_to(point)

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass):
        try:
            self.sign
        except AttributeError:
            self.set_fill(self.get_color(), opacity = np.clip(mass / max(self.initial_mass_list) * self.max_fill_opacity, 0, 1))
        else:
            self.remove(self.sign)
            self.set_fill(self.get_color(), opacity = np.clip(mass / max(self.initial_mass_list) * self.max_fill_opacity, 0, 1))
            self.add(self.sign)
        finally:
            self._mass = mass

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, charge):
        self._charge = charge
        if sum(self.initial_charge_magnitude_list) == 0:
            stroke_width = 0
        else:
            stroke_width = abs(charge) / max(self.initial_charge_magnitude_list) * self.max_stroke_width

        if charge > self.neutrality_tolerance:
            self.set_color(self.positive_charge_colour)
            self_sign = MathTex(self.positive_charge_sign)
        elif charge < self.neutrality_tolerance:
            self.set_color(self.negative_charge_colour)
            self_sign = MathTex(self.negative_charge_sign)
        else:
            self.set_color(self.zero_charge_colour)
            self_sign = MathTex(self.zero_charge_sign)

        self_sign.scale(3.5 * self._radius)
        if within(-self.neutrality_tolerance, charge, self.neutrality_tolerance, "weak"):
            self_sign.scale(0.75)

        self_sign.set_fill(WHITE, opacity = 0)
        self_sign.set_stroke(WHITE, stroke_width)
        self_sign.move_to(self)

        try:
            self.sign
        except AttributeError:
            self.sign = self_sign
        else:
            self.sign.become(self_sign)

    @property
    def _radius(self):
        return distance(self.position, self.get_start())

    @_radius.setter
    def _radius(self, _radius):
        scale_ratio = _radius / self._radius
        self.scale(scale_ratio)

    def has_collided(self, other): # self.has_collided(self) = False
        center_distance = distance(self.position, other.position)

        if other is self:
            collision_status = False
        else:
            if center_distance < self._radius + other._radius:
                collision_status = True
            else:
                collision_status = False

        return collision_status

    def has_crossed_line(self, line):
        line_vector = line.get_end() - line.get_start()
        particle_center_to_line_vector = self.position - line.get_start()
        arbitary_matrix = np.vstack((line_vector, particle_center_to_line_vector, np.array([1, 1, 1])))
        return abs(det(arbitary_matrix)) <= vec_len(line_vector) * self._radius

    def calculate_force_on_particle(self):
        q1 = self.charge
        force = np.array([0, 0, 0], dtype = "float64")
        for other in self.group_of_particles:
            if other is not self:
                q2 = other.charge
                r = distance(self.position, other.position)
                force += (self.attraction_constant * q1 * q2 / r ** 3) * np.array(self.position - other.position, dtype = "float64")
                if self.has_collided(other):
                    repulsive_force = self.collision_force_constant * ((self.mass + other.mass) / 2) * np.array(unit_vec(self.position - other.position), dtype = "float64")
                    force += repulsive_force
        return force

    def calculate_velocity_after_collision(self, other):
        m1 = self.mass
        u1 = np.array(self.velocity, dtype = "float64")
        m2 = other.mass
        u2 = np.array(other.velocity, dtype = "float64")
        p1 = other.position - self.position
        p2 = -1 * p1

        def calculate_unidirectional_velocity(m1, n_u1, m2, n_u2, n_p1):
            u1_m = vec_len(n_u1)
            u2_m = vec_len(n_u2)
            if not in_same_direction(n_u1, n_u2):
                u2_m *= -1
            if in_same_direction(n_u1, n_p1):
                will_collide = True
            else:
                will_collide = False

            if will_collide:
                velocity_value = ((m1 - m2) * u1_m + 2 * m2 * u2_m) / (m1 + m2)
                new_velocity = velocity_value * unit_vec(n_u1)
            else:
                new_velocity = n_u1

            return new_velocity

        u1_along_p1 = (p1.dot(u1) / vec_len(p1) ** 2) * p1
        u1_perpendicular_to_p1 = np.array(u1, dtype = "float64") - np.array(u1_along_p1, dtype = "float64")

        u2_along_p2 = (p2.dot(u2) / vec_len(p2) ** 2) * p2

        u1_along_p1_after_collision = calculate_unidirectional_velocity(m1, u1_along_p1, m2, u2_along_p2, p1)

        v1 = np.array(u1_along_p1_after_collision, dtype = "float64") + np.array(u1_perpendicular_to_p1, dtype = "float64")
        v2 = m1 * (u1 - v1) / m2 + u2

        return (v1, v2)

    def update_position_velocity_and_charge(self, dt, list_of_lines):

        def do_after_collision_between_particles(self, other):
            self.velocity, other.velocity = self.calculate_velocity_after_collision(other)
            if self.is_conductive and other.is_conductive:
                if self.charge != other.charge:
                    self.charge = other.charge = (self.charge + other.charge) / 2
                    self.show_spark((self.position + other.position) / 2, os.path.dirname(__file__) + "/assets/sounds/spark.mp3", color = YELLOW, num_lines = 20)

        if self.consider_collision:
            for other in self.group_of_particles:
                if self.has_collided(other):
                    do_after_collision_between_particles(self, other)

        for line in list_of_lines:
            if self.has_crossed_line(line):
                self.velocity = self.calculate_updated_velocity_after_colliding_with_line(line)

        force = self.calculate_force_on_particle()
        acceleration = np.array(force / self.mass, dtype = "float64") + np.array(self.g * DOWN, dtype = "float64")
        self.velocity = np.array(self.velocity, dtype = "float64") + np.array(acceleration * dt, dtype = "float64")

        self.shift(self.velocity * dt)

    def calculate_updated_velocity_after_colliding_with_line(self, line):
        p1 = line.get_end() - line.get_start()
        u1 = self.velocity
        u1_along_p1 = (p1.dot(u1) / vec_len(p1) ** 2) * p1
        u1_perpendicular_to_p1 = np.array(u1, dtype = "float64") - np.array(u1_along_p1, dtype = "float64")
        arbitary_angle = np.arccos(u1_perpendicular_to_p1.dot(line.get_center() - self.get_center()) / (vec_len(u1_perpendicular_to_p1) * vec_len(line.get_center() - self.get_center())))
        if arbitary_angle < PI / 2:
            new_velocity = np.array(u1_along_p1, dtype = "float64") - np.array(u1_perpendicular_to_p1, dtype = "float64")
            random_area = abs(det(np.vstack((p1, self.position - line.get_start(), np.array([1, 1, 1])))))
            collision_coordinate = random_area / vec_len(p1) * unit_vec(u1_perpendicular_to_p1) + self.position
            self.show_spark(collision_coordinate, os.path.dirname(__file__) + "/assets/sounds/clack.wav", color = self.get_color(), vector = -1 * u1_perpendicular_to_p1, angle_range = PI / 2, num_lines = 10)
        else:
            new_velocity = u1

        return new_velocity

    def show_spark(self, spark_coordinate, spark_sound, **kwargs):
        self.scene.add_sound(spark_sound)
        turn_animation_into_updater(Flash(spark_coordinate, remover = True, rate_func = linear, run_time = 0.4, **kwargs), scene = self.scene, delete_mob = True)

class Testing(Scene):
    CFG = {
    "add_particle_position_updater" : True,
    "rectangle" : Rectangle(height = config.frame_height - 0.5, width = config.frame_width - 0.5),
    "run_time_with_rectangle" : 25,
    "run_time_without_rectangle" : 5,
    "camera_scale_factor" : 1.05,
    "include_vector_field" : True,
    "colour_combination_of_vector_field" : [BLUE_E, GREEN, YELLOW, RED],
    }

    def construct(self):
        self.__dict__.update(self.CFG)
        if self.include_vector_field:
            vec_field = VectorField(Particle.calculate_electric_field_on_point, colors = self.colour_combination_of_vector_field)
            vec_field.add_updater(lambda mob : mob.become(VectorField(Particle.calculate_electric_field_on_point, colors = self.colour_combination_of_vector_field)))
            self.add(vec_field)

        list_of_particles = [
            (2, 1, 6 * LEFT + 3 * UP),
            (6, -2, 3 * LEFT + 2 * DOWN),
            (3, 0, 5 * RIGHT + 3 * UP),
            (4, 3, 6 * RIGHT + 3 * DOWN),
            (1, -3, 2 * LEFT),
            (2, 6, 3 * RIGHT),
            (6, 0, 3 * UP),
            (2, -2, 3 * DOWN + 2 * RIGHT),
            (4, -3, 6 * LEFT),
        ]

        Particle.create_objects([[self] + list(particle_details) for particle_details in list_of_particles], setup = False)

        # print(Particle.get_sum_of_charges())

        for particle in Particle.group_of_particles:
            particle._mass *= 10
            # particle._charge = 0

        Particle.setup()

        if self.include_vector_field:
            objects_on_screen = VGroup(vec_field, Particle.group_of_particles)
        else:
            objects_on_screen = Particle.group_of_particles

        # self.camera.set_frame_height(config.frame_height * self.camera_scale_factor)
        # self.camera.set_frame_width(config.frame_width * self.camera_scale_factor)

        self.add(objects_on_screen)
        self.add(self.rectangle)

        if self.add_particle_position_updater:
            Particle.add_particle_updater(self.rectangle)

        self.wait(self.run_time_with_rectangle)
        Particle.group_of_particles.clear_updaters()
        self.remove(self.rectangle)

        if self.add_particle_position_updater:
            Particle.add_particle_updater()

        self.wait(self.run_time_without_rectangle)