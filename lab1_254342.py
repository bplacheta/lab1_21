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

def generate_points_flat_vertical(num_points: int = 2000, width: float = 100, height: float = 100, length: float = 1,
                                  filename: str = "points.txt"):
    distribution_x = norm(loc=0, scale=width)
    distribution_z = norm(loc=0, scale=height)

    x = distribution_x.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)
    y = np.zeros(num_points)

    points = np.column_stack((x, y, z))

    np.savetxt(filename, points, fmt='%.6f', delimiter=',', header='x,y,z', comments='')

    return points

def generate_points_cylindrical_surface(num_points: int = 2000, radius: float = 50, height: float = 100,
                                        filename: str = "points.txt"):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    points = np.column_stack((x, y, z))

    np.savetxt(filename, points, fmt='%.6f', delimiter=',', header='x,y,z', comments='')

    return points

cloud_points_horizontal = generate_points_flat_horizontal(2000, width=100, length=100, height=1, filename="punkty_płaska_pozioma.xyz")
cloud_points_vertical = generate_points_flat_vertical(2000, width=100, height=100, length=1, filename="punkty_płaska_pionowa.xyz")
cloud_points_cylindrical = generate_points_cylindrical_surface(2000, radius=50, height=100, filename="punkty_cylindryczna.xyz")
