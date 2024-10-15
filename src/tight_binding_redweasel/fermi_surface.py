# all sorts of functions related to the fermi surface
# TODO the plot functions are not be stabilized yet

# plot the fermi surface
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def fermi_surface(model: Callable, fermi_energy: float, k_smpl):
    """Compute the Fermi-surface as a non-normalized signed distance field.

    Args:
        model (Callable): The model to be evaluated. It will be called like a function with k_smpl.
        fermi_energy (float): The Fermi-energy of the system
        k_smpl (_type_): The k-space samples for which the surface will be computed

    Returns:
        (ndarray(), ndarray()): the non-normalized signed distance fields for the bands and the band indices
    """
    shape = np.shape(k_smpl)
    la = model(np.reshape(k_smpl, (-1, 3))) - fermi_energy
    volumes = []
    indices = []
    for i in range(len(la[0])):
        if np.any(la[:, i] > 0) and np.any(la[:, i] < 0):
            volumes.append(np.reshape(la[:, i], shape[:-1]))
            indices.append(i)
    return volumes, indices


def plot_3D_fermi_surface_to_ax(ax, model, fermi_energy, N=32, elev=35, azim=20, k_range=[-0.5, 0.5]):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib
    import matplotlib.colors as mcolors
    from skimage import measure
    x, y, z = np.meshgrid(*3*[np.linspace(*k_range, N)], indexing='ij')
    xyz = np.stack([x, y, z], axis=-1)

    # define light source
    ls = mcolors.LightSource(azdeg=120.0, altdeg=50.0)

    # use default colors from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = list(prop_cycle.by_key()['color'])

    all_verts = []
    all_rgb = []
    volumes, band_indices = fermi_surface(model, fermi_energy, xyz)
    for volume, band, color in zip(volumes, band_indices, colors):
        verts, faces, normals, _ = measure.marching_cubes(volume, 0)

        rgb = ls.shade_normals(normals, 1.0)[..., None] * mcolors.to_rgb(color)
        rgb = rgb[faces]
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        all_verts.extend(verts[faces])
        all_rgb.extend(rgb)
        # TODO add a legend label with {band}
    all_verts = np.array(all_verts)  # shape = (-1, 3, 3)
    # for some reason the matplotlib renders leave gaps between the triangles...
    # Since that is highly annoying, I'm fixing it here by scaling all triangles by a little bit.
    all_verts += 3e-2 * (all_verts - np.mean(all_verts, axis=1, keepdims=True))
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    mesh = Poly3DCollection(
        all_verts/N * (k_range[1] - k_range[0]) + k_range[0], antialiased=False, linewidth=0, linestyle='None')
    mesh.set_edgecolor(None)
    mesh.set_facecolor(np.mean(all_rgb, axis=1))
    # TODO RendererBase.draw_gouraud_triangles exists
    # I can draw vertex colors properly with it,
    # however only tripcolor seems to be the only method using it, which is 2D.
    # -> implement my own copy of Poly3DCollection with gouraud shading.
    ax.add_collection3d(mesh)
    # added view_margin because apparently there is a bug in matplotlib 3.8.3 where it causes a KeyError if it's None.
    ax.set_xlim(*k_range, view_margin=0)
    ax.set_ylim(*k_range, view_margin=0)
    ax.set_zlim(*k_range, view_margin=0)
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)


def plot_3D_fermi_surface(model, fermi_energy, N=32, elev=35, azim=20, k_range=[-0.5, 0.5]):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    plot_3D_fermi_surface_to_ax(
        ax, model, fermi_energy, N, elev, azim, k_range)

    plt.tight_layout(pad=0)
    plt.show()


def export_3D_fermi_surface(file, model, fermi_energy, N=32, k_range=[-0.5, 0.5]):
    from skimage import measure
    import matplotlib.colors as mcolors

    x, y, z = np.meshgrid(*3*[np.linspace(*k_range, N)], indexing='ij')
    xyz = np.stack([x, y, z], axis=-1)

    # use default colors from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = list(prop_cycle.by_key()['color'])

    # could also use OpenUSD, but that requires heavy dependencies... see https://openusd.org/docs/Hello-World---Creating-Your-First-USD-Stage.html
    # so use the simplest obj files possible.

    # first define the materials from the colors above
    with open(f"{file}.mtl", "w") as f:
        f.write("# Material file with matplotlib default colors\n")
        for i, col in enumerate(colors):
            rgb = np.array(mcolors.to_rgb(col))**2.2
            f.write(f"newmtl color{i}\n")
            f.write(f"""Ns 250.000000
Ka 1.000000 1.000000 1.000000
Kd {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
""")

    # now create the obj file with the data
    with open(f"{file}.obj", 'w') as f:
        f.write("# OBJ file from tight_binding_redweasel.fermi_surface\n")
        f.write(f"mtllib {file}.mtl\n")
        f.write(f"o {file}\n")
        f.write(f"s 1\n")

        volumes, band_indices = fermi_surface(model, fermi_energy, xyz)
        vert_index = 0
        for index, (volume, band) in enumerate(zip(volumes, band_indices)):
            verts, faces, normals, _ = measure.marching_cubes(volume, 0)
            # create objects for the bands
            f.write(f"g Band{band}\n")
            for v in verts:
                f.write("v %.4f %.4f %.4f\n" % tuple(v))
            f.write(f"usemtl color{index}\n")
            for face in faces:
                f.write("f")
                for i in face:
                    f.write(" %d" % (i + 1 + vert_index))
                f.write("\n")
            vert_index += len(verts)


# plot two cuts of the 3D fermi surface at k=(x,y,z) with given z
def plot_2D_fermi_surface(model, fermi_energy, z=[0, 1/2], N=50, show=plt.show, k_range=[-0.5, 0.5]):
    x_ = np.linspace(*k_range, N)
    y_ = np.linspace(*k_range, N)
    z_ = np.array([0.0])
    x, y, z_ = np.meshgrid(x_, y_, z_, indexing='ij')

    for z2 in z:
        z_ = z_*0.0 + z2
        k_smpl = np.stack([x, y, z_], axis=-1)
        shape = list(np.shape(k_smpl))
        la = model(np.reshape(k_smpl, (-1, 3)))
        color_index = 0
        for i in range(len(la[0])):
            volume = np.reshape(la[:, i], shape[:-1]) - fermi_energy
            if np.any(volume > 0) and np.any(volume < 0):
                cs = plt.contour(x.reshape((N, N)), y.reshape(
                    (N, N)), volume.reshape((N, N)), (0,), colors=f"C{color_index}")
                # plt.clabel(cs, inline=1, fontsize=10) # for the case where more than 1 line is plotted
                for c in cs.collections:
                    c.set_label(f"Band {i}")
                # print(i)
                color_index += 1
        # set limits for the case, that there is no bands plotted
        plt.xlim(k_range[0], k_range[1])
        plt.ylim(k_range[0], k_range[1])
        plt.gca().set_aspect("equal")
        plt.title("(001)-plane cut")
        # plt.legend(loc="lower right") # doesn't work... plots really weird lines instead of just colors
        if show:
            show()
