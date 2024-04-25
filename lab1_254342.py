import numpy as np
from scipy.stats import norm


def generate_points_flat_horizontal(num_points: int = 2000, width: float = 100, length: float = 100, height: float = 1,
                                    filename: str = "points.txt"):
    distribution_x = norm(loc=0, scale=width)
    distribution_y = norm(loc=0, scale=length)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = np.random.uniform(0, height, num_points)

    points = np.column_stack((x, y, z))

    np.savetxt(filename, points, fmt='%.6f', delimiter=',', header='x,y,z', comments='')

    return points


cloud_points = generate_points_flat_horizontal(2000, width=100, length=100, height=1, filename="punkty.xyz")
