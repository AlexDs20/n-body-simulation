import numpy as np

class object():
    def __init__(self, init_pos, init_vel, mass, radius):
        if len(init_pos) == 3:
            self.pos = np.array(init_pos)
        else:
            return None
        if len(init_vel) == 3:
            self.vel = np.array(init_vel)
        else:
            return None
        self.mass = mass
        self.radius = radius

    def __str__(self):
        return f'{self.pos}\t--\t{self.vel}'

    def speed(self):
        return np.sqrt(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)


class universe():
    # The General equations:
    # =====================
    #
    # Newton:
    # ------
    #   F_i/m_i = d^2x_i/dt^2 = dv_i/dt
    #
    # Force on ith obj:
    # ----------------
    #   F_i/m_i = G * \sum_{j\diff i} (m_j / r_ij^2)  * \vec{r}_ij/r_ij
    #
    # Equation for the ith body:
    # ------------------------
    #   1) Get the new velocity
    #       dv_i/dt                     =   G * \sum_{j\diff i} (m_j / r_ij^2)  * \vec{r}_ij/r_ij
    #       (v_i(t+1) - vi(t)) / dt     =   G * \sum_{j\diff i} (m_j / r_ij^2)  * \vec{r}_ij/r_ij
    # --->  v_i(t+1)                    =   v_i(t) + dt * (G * \sum_{j\diff i} (m_j / r_ij^2)  * \vec{r}_ij/r_ij)
    #
    #   2) Update the position
    #       dx_i/dt = v_i
    #       (x_i(t+1) - x_i(t))/dt      =   v_i
    # --->  x_i(t+1)                    =   x_i(t) + dt * v_i

    G = 6.6743*10**-11
    dx_prop = 1/20


    def __init__(self, objects):
        # All the planets
        self.objects = objects

        # The "Force matrix"
        self.F = np.zeros((len(objects), len(objects), 3))

        # Get the maximal spatial step from the radius of the objects
        self.max_dx = min(o.radius for o in objects) * self.dx_prop

        # Set a first time step
        self.dt = self.calc_time_step()



    def print(self):
        for i, o in enumerate(self.objects):
            print(f'object {i}:\t', o, '\n')


    def calc_all_forces(self):
        def calc_r_divided_r3(o1, o2):
            diff = o2.pos - o1.pos
            denum = (diff[0]**2 + diff[1]**2 + diff[2]**2)**(3/2)
            return diff/denum

        for i in range(len(self.objects)):
            for j in range(i+1, len(self.objects)):
                f_over_gmm = calc_r_divided_r3(self.objects[i], self.objects[j])
                self.F[i, j] = f_over_gmm
                self.F[j, i] = - self.F[i, j]


    def calc_time_step(self):
        # Given the current velocities and the max spatial steps, update the time step
        # Get the current max speed
        max_speed = max(o.speed() for o in self.objects)
        return self.max_dx / max_speed


    def update_velocities(self):
        # --->  v_i(t+1) = v_i(t) + dt * (G * \sum_{j\diff i} (m_j / r_ij^2)  * \vec{r}_ij/r_ij)
        for i, ref in enumerate(self.objects):
            # Calculate force acting on object i
            #   -> sum the contributions from all the others

            sum_forces = 0
            for j, o in enumerate(self.objects):
                sum_forces += self.G * o.mass * self.F[i, j, :]

            ref.vel += self.dt * sum_forces


    def update_positions(self):
        # --->  x_i(t+1)                    =   x_i(t) + dt * v_i
        for o in self.objects:
            o.pos += self.dt * o.vel

    def is_collision(self):
        pass

    def merge_if_collision(self):
        def calc_center_of_mass(self):
            pass
        def calc_new_vel(self):
            pass
        def calc_new_mass(self):
            pass

    def run(self, time):
        t = 0

        while t<time:
            self.calc_all_forces()
            self.update_velocities()
            self.calc_time_step()
            self.update_positions()
            self.print()
            t = t + self.dt


def main():
    o1 = object([0. ,     0.    ,   0.], \
                [0. ,     0.5   ,   0.], \
                3, \
                0.1)

    o2 = object([1. ,     1.    ,   0.], \
                [0. ,     0.5   ,   0.], \
                0.1, \
                4)

    uni = universe([o1, o2])
    uni.run(time=10)

if __name__=='__main__':
    main()
