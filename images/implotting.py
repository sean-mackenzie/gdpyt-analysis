import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

plt.style.use(['science', 'ieee', 'std-colors'])
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


def plot_gaussian_3d(x, y, z):
    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax


def plot_image_and_gaussian_3d(x, y, z1, z2):
    """ fig, (ax1, ax2) = plot_image_and_gaussian_3d(x, y, z1, z2) """
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # set up the axes for the first plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x, y, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # set up the axes for the second plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x, y, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    return fig, (ax1, ax2)


def plot_image_and_gaussian_2d(z1, z2):
    """ fig, (ax1, ax2) = plot_image_and_gaussian_2d(z1, z2) """

    fig, (ax1, ax2) = plt.subplots(figsize=(size_x_inches, size_y_inches), ncols=2,
                                   gridspec_kw={'width_ratios': [1, 1]},
                                   )

    # first plot
    p1 = ax1.imshow(z1, cmap='magma', interpolation='none')
    # cbar1 = fig.colorbar(p1, ax=ax1, location='left', extend='both', shrink=0.45, label='Pixel value (A.U.)')
    # cbar1.minorticks_on()

    # Plot both positive and negative values between +/- 1.2
    p2 = ax2.imshow(z2, cmap='RdBu', interpolation='none')
    # cbar2 = fig.colorbar(p2, ax=ax2, location='right', extend='both', shrink=0.45, label='Residuals (A.U.)')
    # cbar2.minorticks_on()

    ax1.tick_params(axis='both', which='both',
                    left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False, top=False)
    ax2.tick_params(axis='both', which='both',
                    left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False, top=False)

    # adjust subplots to make room for color bars
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.9)

    # get position and size of images on figure
    ax1_size = ax1.axes.figbox.bounds
    ax2_size = ax2.axes.figbox.bounds

    # create new axes for color bars
    cax1 = plt.axes([ax1_size[0] - 0.075, ax1_size[1], 0.02, ax1_size[3]])
    cax2 = plt.axes([ax2_size[0] + ax2_size[2] + 0.0525, ax2_size[1], 0.02, ax2_size[3]])

    # create color bars
    cbar1 = fig.colorbar(p1, cax=cax1, extend='both', aspect=50, ticklocation='left')
    cbar1.minorticks_on()
    cbar1.set_label(label='Pixel values (A.U.)')

    cbar2 = fig.colorbar(p2, cax=cax2, extend='both', aspect=50, label='Residuals (A.U.)')
    cbar2.minorticks_on()

    # plt.tight_layout()
    return fig, (ax1, ax2)


def plot_point_spread_function(data, focal_plane_diff=False):
    if isinstance(data, dict):
        img_stack = [arr for key, arr in data.items()]
        img_stack = np.array(img_stack)
    elif isinstance(data, list):
        img_stack = [arr for arr in data]
        img_stack = np.array(img_stack)
    elif isinstance(data, np.ndarray):
        img_stack = data
    else:
        raise ValueError("data type not understood.")


    slice_idx = int(np.floor(np.shape(img_stack)[1] / 2))

    if not focal_plane_diff:
        img_zx = img_stack[:, slice_idx, :]
        fig, ax = plt.subplots(figsize=(size_x_inches / 2, size_y_inches * 2))
        ax.imshow(img_zx, cmap='RdBu', interpolation='antialiased')
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_xlabel(r'$x \: (\mu m)$')
        plt.tight_layout()
    else:
        img_norm_zx_zy = img_stack[:50, slice_idx, :] - np.flip(img_stack[50:100, slice_idx, :], axis=0)
        fig, ax = plt.subplots(figsize=(size_x_inches / 1.5, size_y_inches))
        p = ax.imshow(img_norm_zx_zy, cmap='RdBu', interpolation='antialiased')
        cbar2 = fig.colorbar(p, ax=ax, extend='both', shrink=0.85)
        cbar2.minorticks_on()
        ax.set_ylabel(r'$z \: (\mu m)$')
        ax.set_xlabel(r'$x \: (\mu m)$')
        plt.tight_layout()

    return fig, ax