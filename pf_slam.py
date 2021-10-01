from utils.visualize import motion_plot, scatter_plot, motion_lidar_plot
from motion_models.differential_drive_models import discrete_time_model
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# string constants for file parsing
ODOM = "ODOM"
FLASER = "FLASER"
PARAM = "PARAM"

# num angles
ANGLE_RANGE = 360

# particle constants
# NUM_PARTICLES = 1000
NUM_PARTICLES = 200
THRES = NUM_PARTICLES / 10
MAX_RANGE = 80

# world constants
EXTEND_AREA = 0
# GRID_SIZE = 0.2
GRID_SIZE = 0.5
RISE = 0

# noise constants
# R = np.diag([0.001, 0.000001])  # input error low
R = np.diag([0.00095, 0.01])  # input error best
# R = np.diag([0, 0]) ** 2  # input error no

# confidence to use when calculating log odds
FREE_CONFIDENCE = 0.4
OCCUPIED_CONFIDENCE = 0.8
MAX_LOG_VAL = 5
MIN_LOG_VAL = -5
CONFIDENCE_VAL = 2


class Particle:

    def __init__(self, pose, weight):
        self.pose = pose
        self.weight = weight
        self.color = None


def particle_filter(particles, grid_map, log_odd_map, binary_map, sensor_ranges, sensor_angles, dims, k):
    num_grid_cells, grid_width, grid_height, minx, miny, maxx, maxy = dims
    x = []
    y = []
    t = []

    # sample particles
    for i in range(NUM_PARTICLES):
        x.append(particles[i].pose[0][0])
        y.append(particles[i].pose[1][0])
        t.append(particles[i].pose[2][0])
    x = np.stack(x).reshape(-1, 1)
    y = np.stack(y).reshape(-1, 1)
    t = np.stack(t).reshape(-1, 1)

    # calc observations according to the sample poses
    p_x = sensor_ranges * np.cos(t + sensor_angles) + x
    p_y = sensor_ranges * np.sin(t + sensor_angles) + y

    # calculate binary map
    binary_map.fill(0)
    indices = log_odd_map > CONFIDENCE_VAL
    binary_map[indices] = 1

    # calc map correlation for each particle
    corrs = calc_map_corrs(particles, binary_map, minx, maxx, miny, maxy, p_x, p_y, k)
    probs = softmax_prob(corrs)

    # update weights for each particle
    for i in range(NUM_PARTICLES):
        particles[i].weight  = particles[i].weight * probs[i]

    normalize(particles)

    # calc new map according to highest weighted particle
    weights = np.array([p.weight for p in particles])
    idx = np.argmax(weights)

    grid_map, log_odd_map = set_grid_map(particles[idx], p_x[idx], p_y[idx], grid_map, log_odd_map, grid_width, grid_height, minx, miny)

    # calc neff
    neff = 0
    for p in particles:
        neff += p.weight ** 2
    neff = 1 / neff

    # if neff is less than a threshold, resample
    if neff < THRES:
        particles = low_variance_sampler(particles)
    return particles, grid_map, log_odd_map


def normalize(particles):
    total = 0
    for p in particles:
        total += p.weight
    for p in particles:
        p.weight = p.weight / total


def low_variance_sampler(particles):
    new_particles = []
    r = random.uniform(0, 1/NUM_PARTICLES)
    c = particles[0].weight # get first weight
    i = 0

    for m in range(NUM_PARTICLES):
        u = r + m * (1 / NUM_PARTICLES)
        while u > c:
            i += 1
            c += particles[i].weight
        new_particles.append(Particle(particles[i].pose, 1 / NUM_PARTICLES))
    return np.array(new_particles)


def calc_map_corrs(particles, binary_map, minx, maxx, miny, maxy, p_x, p_y, k):
    # calc map corr for each particle
    corrs = []
    for i in range(len(particles)):
        # offsets used for map corr
        xs = np.array([-0.4, -0.2, 0, 0.2, 0.4])
        ys = np.array([-0.4, -0.2, 0, 0.2, 0.4])

        c = mapCorrelation(binary_map, (minx,maxx), (miny,maxy), np.vstack((p_x[i],p_y[i])), xs, ys, k)
        corrs.append(c.flatten()[np.argmax(c)])
        row = np.argmax(c) % c[0].shape[0]
        col = int(np.argmax(c) / c[0].shape)

        # alter particle position by best map corr
        particles[i].pose[0] += xs[col]
        particles[i].pose[1] += ys[row]

    return np.array(corrs)


def mapCorrelation(im, x_im, y_im, vp, xs, ys, k):
    '''
    INPUT
    im              the map
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)
    xs,ys           physical x,y,positions you want to evaluate "correlation"

    OUTPUT
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]

    xmin = x_im[0]
    xmax = x_im[-1]

    ymin = y_im[0]
    ymax = y_im[-1]

    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))

    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16((y1 - ymin) / GRID_SIZE)

        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076

            ix = np.int16((x1 - xmin) / GRID_SIZE)

            cpr[jx,jy] = np.sum(im[ix,iy])

    return cpr


def softmax_prob(corrs):
    e = np.exp(corrs)
    total = np.sum(e)
    return e / total


def move_particles(particles, control, dt):
    for i in range(NUM_PARTICLES):
        c =  [control[0] + np.random.randn() * R[0, 0] ** 0.5, control[1] +
              np.random.randn() * R[1, 1] ** 0.5]
        particles[i].pose = discrete_time_model(particles[i].pose, c, dt)


def inverse_sensor_model(occupied):
    if occupied:
        return math.log(OCCUPIED_CONFIDENCE / (1 - OCCUPIED_CONFIDENCE))
    else:
        return math.log(FREE_CONFIDENCE / (1 - FREE_CONFIDENCE))


def dist(x1, y1, x2, y2):
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def set_grid_map(particle, end_x, end_y, grid_map, log_odd_map, grid_width, grid_height, minx, miny):
    # translate start position
    start_x = int(round((particle.pose[0][0] - minx) / GRID_SIZE))
    start_y = int(round((particle.pose[1][0] - miny) / GRID_SIZE))

    # update grid map based on each lidar endpoint
    for e_x, e_y in zip(end_x, end_y):
        # filter out maximum range
        if dist(particle.pose[0][0], particle.pose[1][0], e_x, e_y) < MAX_RANGE:
            # translate endpoint
            x = int(round((e_x - minx) / GRID_SIZE))
            y = int(round((e_y - miny) / GRID_SIZE))

            # get occupied pixels
            pixels = bresenham((start_x, start_y), (x, y))

            # update log odd for free cells
            for i in range(len(pixels) - 1):
                if pixels[i][0] < grid_width and pixels[i][1] < grid_height:
                    if log_odd_map[pixels[i]] > MIN_LOG_VAL:
                        log_odd_map[pixels[i]] += inverse_sensor_model(False)
                        grid_map[pixels[i]] = 1 - 1/(math.e ** log_odd_map[pixels[i]] + 1)

            if log_odd_map[x][y] < MAX_LOG_VAL:
                log_odd_map[x][y] += inverse_sensor_model(True) # occupied area
                grid_map[x][y] = 1 - 1/(math.e ** log_odd_map[x][y] + 1)

    return grid_map, log_odd_map


def init_grid_map(minx, maxx, miny, maxy):
    grid_width = int(round((maxx - minx) / GRID_SIZE))
    grid_height = int(round((maxy - miny) / GRID_SIZE))
    num_grid_cells = grid_width * grid_height
    print("Num Grid Cells: " + str(num_grid_cells) + " Grid Width: " + str(grid_width) + " Grid Height: " + str(grid_height))
    grid_map = np.full((grid_width, grid_height), 0.5)
    return grid_map, grid_width, grid_height, num_grid_cells


def bresenham(start, end):
    """ returns the shortest path through the pixel grid that connects the start and end points """
    points = []
    i_ind = 0
    j_ind = 1
    inc = 1
    swap = False
    reverse = False

    # determine whether to iterate over x or y
    if abs(start[0] - end[0]) < abs(start[1] - end[1]):
        i_ind = 1
        j_ind = 0
        swap = True

    # determine the start and end points
    if start[i_ind] > end[i_ind]:
        temp = end
        end = start
        start = temp
        reverse = True

    # calc deltas
    d_j = end[j_ind] - start[j_ind]
    d_i = end[i_ind] - start[i_ind]

    # determine if j increases or decreases
    if start[j_ind] > end[j_ind]:
        inc = -1
        d_j = abs(d_j)

    m = None
    if d_i != 0:
        m = d_j / d_i # slope

    if m is None:
        pass
    else:
        j = start[j_ind]
        e = 0
        for i in range(start[i_ind], end[i_ind] + 1):
            if swap:
                points.append((j, i))
            else:
                points.append((i, j))
            if 2 * (e + d_j) < d_i:
                e += d_j
            else:
                j += inc
                e += (d_j - d_i)
        if reverse:
            points.reverse()

    if not points:
        points.append(start)
    return points


def get_lidar_data(path):
    with open(path) as data:
        data = data.readlines()

    # init lists to return
    robot_x = []
    robot_y = []
    robot_t = []
    sensor_ranges = []
    lidar_ts = []
    sensor_angles = [(math.radians((i - 180) * 0.5)) for i in range(ANGLE_RANGE)]

    for line in data:
        d = line.split(' ')

        # get laser pose and ranges
        if d[0] == FLASER:
            robot_x.append(float(d[-9]))
            robot_y.append(float(d[-8]))
            robot_t.append(float(d[-7]))
            sensor_ranges.append(list(map(float, d[2:-9])))
            lidar_ts.append(float(d[-1]))

    return np.stack(robot_x).reshape(-1, 1), np.stack(robot_y).reshape(-1, 1), \
            np.stack(robot_t).reshape(-1, 1), np.stack(sensor_angles), \
            np.stack(sensor_ranges), np.stack(lidar_ts).reshape(-1, 1)


def get_motion_data(path):
    with open(path) as data:
        data = data.readlines()

    # init lists to be returned
    dr_data = []
    odom_data = []
    dt_data = []
    motion_ts = []

    prev_t = 0
    for i in range(len(data)):
        d = data[i].split(' ')
        n = None
        if i + 1 < len(data):
            n = data[i + 1].split(' ')

        # get odom and dead reckoning data
        if d[0] == ODOM:
            obs = False
            if n[0] == FLASER:
                obs = True
            dr_data.append([[float(d[1])], [float(d[2])], [float(d[3])]])
            odom_data.append([float(d[4]), float(d[5]), obs])
            motion_ts.append(float(d[-1]))
    return np.stack(dr_data), np.array(odom_data), np.array(dt_data), np.array(motion_ts)


def main():
    path = "data/data_correct.log"
    pause = False

    # get lidar and motion data
    l_robot_x, l_robot_y, l_robot_t, sensor_angles, sensor_ranges, lidar_ts = get_lidar_data(path)
    dr_data, odom_data, dt_data, motion_ts = get_motion_data(path)

    # init pose and pose stack
    pose = dr_data[0]

    # init particles with weights
    particles = np.array([(Particle(dr_data[0], 1 / NUM_PARTICLES)) for _ in range(NUM_PARTICLES)])

    # calc sensor endpoints
    end_x = sensor_ranges * np.cos(l_robot_t + sensor_angles) + l_robot_x
    end_y = sensor_ranges * np.sin(l_robot_t + sensor_angles) + l_robot_y

    minx = math.floor(min(end_x.flatten()) - EXTEND_AREA / 2.0)
    miny = math.floor(min(end_y.flatten()) - EXTEND_AREA / 2.0)
    maxx = math.ceil(max(end_x.flatten()) + EXTEND_AREA / 2.0)
    maxy = math.ceil(max(end_y.flatten()) + EXTEND_AREA / 2.0)

    w_height = maxy - miny
    w_width = maxx - minx
    print("World Width: " + str(w_width) + " World Height: " + str(w_height))

    # get world and grid dimensions
    grid_map, grid_width, grid_height, num_grid_cells = init_grid_map(minx, maxx, miny, maxy)
    log_odd_map = np.zeros((grid_width, grid_height))
    binary_map = np.zeros((grid_width, grid_height))

    grid_map, log_odd_map = set_grid_map(particles[0], end_x[0], end_y[0], grid_map, log_odd_map, grid_width, grid_height, minx, miny)

    dims = num_grid_cells, grid_width, grid_height, minx, miny, maxx, maxy

    # particle filter for every lidar scan
    l_idx = 0
    m_idx = 0
    dt_sum = 0
    while m_idx < len(odom_data) and l_idx < len(lidar_ts):
        # printing and matplotlib display
        if m_idx % 400 == 0:
            plt.figure(2)
            plt.title("Tv: " + str(R[0,0]) + " Rv: " + str(R[1,1]))
            plt.imshow(grid_map)
            xs = np.array([p.pose[0,0] for p in particles])
            ys = np.array([p.pose[1,0] for p in particles])
            xs = np.int16((xs - minx) / GRID_SIZE)
            ys = np.int16((ys - miny) / GRID_SIZE)
            plt.scatter(ys, xs, s=1, c='r')
            plt.pause(0.001)

        if m_idx % 400 == 0:
            print("Time step " + str(m_idx) + " out of " + str(len(odom_data)))

        # when a odom command has a observation, execute the particle filter
        if odom_data[m_idx][2] == True:
            dt = lidar_ts[l_idx][0] - motion_ts[m_idx]
            dt_sum += dt
            move_particles(particles, odom_data[m_idx], dt)

            if l_idx % 3 == 0:
                particles, grid_map, log_odd_map = particle_filter(particles, grid_map, log_odd_map, binary_map, sensor_ranges[l_idx], sensor_angles, dims, m_idx)

            dt = 0
            if m_idx + 1 < len(odom_data):
                dt = motion_ts[m_idx + 1] - lidar_ts[l_idx][0]

            dt_sum += dt
            move_particles(particles, odom_data[m_idx], dt)

            l_idx += 1
            m_idx += 1
        else:
            # calc new poses for particles and odom trajectory according to motion model
            dt = 0
            if m_idx + 1 < len(odom_data):
                dt = motion_ts[m_idx + 1] - motion_ts[m_idx]
            dt_sum += dt

            move_particles(particles, odom_data[m_idx], dt)
            m_idx += 1

    plt.cla()
    plt.imshow(grid_map)
    plt.colorbar()
    plt.pause(0.001)
    plt.show()

if __name__ == "__main__":
    main()
