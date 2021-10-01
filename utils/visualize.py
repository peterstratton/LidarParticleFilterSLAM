import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(xs, ys, area, colors, a=0.5, title=None, x_label=None, y_label=None):
    plt.scatter(np.asarray(xs), np.asarray(ys), s=area, c=colors, alpha=a)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()

def motion_plot(pause, s1, s2=None, s3=None):
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    plt.plot(s1[0, :].flatten(), s1[1, :].flatten(), ".g", alpha=0.5)

    if s2 is not None:
        plt.plot(s2[0, :].flatten(), s2[1, :].flatten(), ".b", alpha=0.5)

    if s3 is not None:
        plt.plot(s3[0, :].flatten(), s3[1, :].flatten(), ".r", alpha=0.5)

    plt.axis("equal")
    plt.grid(True)

    if pause:
        plt.pause(0.001)
    else:
        plt.show()

def motion_lidar_plot(pause, s1, s2, end_x, end_y, area, colors, a=0.5):
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    plt.scatter(end_x, end_y, s=area, c=colors, alpha=a)

    plt.plot(s1[0, :].flatten(), s1[1, :].flatten(), ".g", alpha=a)

    if s2 is not None:
        plt.plot(s2[0, :].flatten(), s2[1, :].flatten(), ".b", alpha=a)

    plt.axis("equal")
    plt.grid(True)

    if pause:
        plt.pause(0.001)
    else:
        plt.show()
