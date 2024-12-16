import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

class Track:
    def __init__(self, ctrl_points, centerline, centerline_curv, centerline_yaw, interior, exterior, cones_interior, cones_exterior):
        self.ctrl_points = ctrl_points
        self.centerline = centerline
        self.centerline_curv = centerline_curv
        self.centerline_yaw = centerline_yaw
        self.interior = interior
        self.exterior = exterior
        self.cones_interior = cones_interior
        self.cones_exterior = cones_exterior

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.ctrl_points[0], self.ctrl_points[1], 'o', label='Control Points', color='red')
        # plt.plot(self.centerline[0], self.centerline[1], '-', label='Centerline Spline', color='blue')
        plt.scatter(self.centerline[0], self.centerline[1], c=self.centerline_curv, cmap='jet', s=5, label='Centerline')  # Using a color map
        plt.colorbar(label="Curvature")  # Show a color bar to indicate curvature levels
        plt.plot(self.interior[0], self.interior[1], '-', label='Interior TLs', color='green')
        plt.plot(self.exterior[0], self.exterior[1], '-', label='Exterior TLs', color='darkgreen')
        plt.plot(self.cones_interior[0], self.cones_interior[1], '^', label='Interior Cones', color='brown')
        plt.plot(self.cones_exterior[0], self.cones_exterior[1], '^', label='Exterior Cones', color='red')
        plt.axis('equal')  # Equal scaling
        plt.legend()
        plt.title('Your generated FS Track')
        plt.grid(True)
        plt.show()

class TrackGenerator:
    def __init__(self, n_points, radius_m, noise_deform_m, smooth_factor, track_width_m, cone_spacing_m, cone_position_std, max_curvature=1/9.0, max_iters=10, sampling_num=500):
        self.n_points = n_points
        self.noise_deform_m = noise_deform_m
        self.max_curvature = max_curvature
        self.max_iters = max_iters
        self.smooth_factor = smooth_factor
        self.track_width_m = track_width_m
        self.cone_spacing_m = cone_spacing_m
        self.cone_position_std = cone_position_std
        self.sampling_num = sampling_num
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)  # Angles for circle
        self.x_circle = radius_m * np.cos(theta)  # x-coordinates of the circle
        self.y_circle = radius_m * np.sin(theta)  # y-coordinates of the circle

    def _spline_length(self, tck, t) -> float:
        """
        Computes the length of a periodic spline.
        """
        dx_dt, dy_dt = splev(t, tck, der=1)
        integrand = np.sqrt(dx_dt**2 + dy_dt**2)
        return np.trapz(integrand, t)


    def _compute_curvature(self, tck, t) -> np.ndarray:
        """
        Given a spline, computes the curvature evaluated at the points given by t.

        Input:
        - tck: a spline tuple
        - t: array of points at which the spline will be evaluated

        Output:
        - array containing the curvature of the spline evaluated at points t
        """
        dx_dt, dy_dt = splev(t, tck, der=1)
        d2x_dt2, d2y_dt2 = splev(t, tck, der=2)
        return (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    def _compute_yaw(self, tck, t) -> np.ndarray:
        """
        Given a spline, computes the yaw angles evaluated at the points given by t.

        Input:
        - tck: a spline tuple
        - t: array of points at which the spline will be evaluated

        Output:
        - array containing the yaw angles of the spline evaluated at points t
        """
        dx_dt, dy_dt = splev(t, tck, der=1)
        return np.arctan2(dy_dt, dx_dt)

    def _generate_centerline(self, ctrl_points) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Takes the control points and generates the centerline by interpolating a spline.

        Input:
        - ctrl_points: tuple containing the x and y coords of the control points (2,n)

        Output:
        - centerline: tuple containing the x and y coords of the centerline (2,m)
        - curvature: contains the curvature of every point in the centerline (m)
        - yaw: contains the yaw angle of every point in the centerline (m)
        """
        # Fit parametric splines (t -> x and t -> y)
        tck, u = splprep(ctrl_points, per=1, s=self.smooth_factor)  # Periodic spline for x

        t_dense = np.linspace(0, 1, 500)  # Dense t for smooth curve

        centerline = splev(t_dense, tck)
        curvature = self._compute_curvature(tck, t_dense)
        yaw = self._compute_yaw(tck, t_dense)

        return centerline, curvature, yaw

    def _generate_tracklimits(self, centerline) -> tuple[list[np.ndarray]]:
        """
        Generates the track limits based on the centerline, it does so by computing the
        normal vectors of each point and translating the point in that direction in
        and out. Then a spline is computed and points are sampled.
        For the cones of the track limits, we put n_cones = spline_length/sample_dist.

        Input:
        - centerline: tuple containing the x and y coords of the centerline (2,m)

        Output:
        - interior: tuple containing the x and y coords of the interior tracklimits (2,i)
        - exterior: tuple containing the x and y coords of the exterior tracklimits (2,j)
        - cones_interior: tuple containing the x and y coords of the interior cones (2,k)
        - exterior: tuple containing the x and y coords of the exterior tracklimits (2,l)
        """
        centerline_x = centerline[0]
        centerline_y = centerline[1]
        t_dense = np.linspace(0, 1, self.sampling_num)
        # Step 1: Compute tangent and normal vectors
        dx = np.gradient(centerline_x, t_dense)
        dy = np.gradient(centerline_y, t_dense)

        # Normalize tangent vectors
        tangent_magnitude = np.sqrt(dx**2 + dy**2)
        tangent_x = dx / tangent_magnitude
        tangent_y = dy / tangent_magnitude

        # Compute normal vectors (perpendicular to tangents)
        normal_x = -tangent_y
        normal_y = tangent_x

        # Step 2: Offset the centerline
        interior_x = centerline_x - (self.track_width_m/2) * normal_x
        interior_y = centerline_y - (self.track_width_m/2) * normal_y
        exterior_x = centerline_x + (self.track_width_m/2) * normal_x
        exterior_y = centerline_y + (self.track_width_m/2) * normal_y

        # Step 3: Fit splines to the interior and exterior limits
        interior_tck, interior_t = splprep([interior_x, interior_y], per=1, s=self.smooth_factor)
        exterior_tck, exterior_t = splprep([exterior_x, exterior_y], per=1, s=self.smooth_factor)

        # Generate dense points for smooth curves
        interior_smooth = splev(t_dense, interior_tck)
        exterior_smooth = splev(t_dense, exterior_tck)

        interior_length = self._spline_length(interior_tck, t_dense)
        exterior_length = self._spline_length(exterior_tck, t_dense)

        interior_cones_t = np.linspace(0, 1, int(interior_length/self.cone_spacing_m))
        exterior_cones_t = np.linspace(0, 1, int(exterior_length/self.cone_spacing_m))

        cones_interior = splev(interior_cones_t, interior_tck)
        cones_exterior = splev(exterior_cones_t, exterior_tck)

        # Add noise to cones
        cones_interior[0] += np.random.normal(0.0, self.cone_position_std, size=cones_interior[0].size)
        cones_interior[1] += np.random.normal(0.0, self.cone_position_std, size=cones_interior[1].size)
        cones_exterior[0] += np.random.normal(0.0, self.cone_position_std, size=cones_exterior[0].size)
        cones_exterior[1] += np.random.normal(0.0, self.cone_position_std, size=cones_exterior[1].size)

        return interior_smooth, exterior_smooth, cones_interior, cones_exterior
    
    def generate(self) -> Track:
        for iter in range(self.max_iters):
            # Deform the circle
            # np.random.seed(48)  # For reproducibility
            x_noise = np.random.uniform(-self.noise_deform_m, self.noise_deform_m, size=self.n_points)  # Small random shifts
            y_noise = np.random.uniform(-self.noise_deform_m, self.noise_deform_m, size=self.n_points)
            x_ctrl_points = self.x_circle + x_noise
            y_ctrl_points = self.y_circle + y_noise

            # Ensure periodicity by appending the first point to the end
            x_ctrl_points = np.append(x_ctrl_points, x_ctrl_points[0])
            y_ctrl_points = np.append(y_ctrl_points, y_ctrl_points[0])
            ctrl_points = [x_ctrl_points, y_ctrl_points]

            # Generate the centerline
            centerline, centerline_curv, centerline_yaw = self._generate_centerline(ctrl_points)
            
            max_curvature = np.max(centerline_curv)
            print(max_curvature)

            # if max_curvature > self.max_curvature:
            #     print("max curvature exceeded")
            #     continue
            
            # Generate the tracklimits & cones
            interior, exterior, cones_interior, cones_exterior = self._generate_tracklimits(centerline)

            return Track(ctrl_points, centerline, centerline_curv, centerline_yaw, interior, exterior, cones_interior, cones_exterior)
        
        print("Cannot generate track with meeting constraints.")

generator = TrackGenerator(n_points=8, radius_m=40.0, noise_deform_m=15.0, smooth_factor=5.0, track_width_m=3.5, cone_spacing_m=5.0, cone_position_std=0.3)

track = generator.generate()
track.plot()