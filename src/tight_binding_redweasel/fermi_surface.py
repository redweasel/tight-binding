# all sorts of functions related to the fermi surface
# TODO the plot functions are not be stabilized yet

# plot the fermi surface
import numpy as np
from matplotlib import pyplot as plt

# get the fermi surface as a signed distance field.
# returns the signed distance fields for the bands and the band indices
def fermi_surface(model, fermi_energy, k_smpl):
    shape = list(np.shape(k_smpl))
    la = model(np.reshape(k_smpl, (-1, 3))) - fermi_energy
    volumes = []
    indices = []
    for i in range(len(la[0])):
        if np.any(la[:,i] > 0) and np.any(la[:,i] < 0):
            volumes.append(np.reshape(la[:,i], shape[:-1]))
            indices.append(i)
    return volumes, indices


def plot_3D_fermi_surface(model, fermi_energy, N=32, elev=35, azim=20, k_range=[-0.5, 0.5]):
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
        rgb = np.mean(rgb[faces], axis=1) # convert to faces
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        all_verts.extend(verts[faces])
        all_rgb.extend(rgb)
        # TODO add a legend label with {band}

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(np.array(all_verts)/N * (k_range[1] - k_range[0]) + k_range[0], antialiased=False)
    mesh.set_edgecolor(None)
    mesh.set_facecolor(all_rgb)
    ax.add_collection3d(mesh)
    ax.set_xlim(*k_range)
    ax.set_ylim(*k_range)
    ax.set_zlim(*k_range)
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
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

    for z__ in z:
        z_ = z_*0.0 + z__
        k_smpl = np.stack([x, y, z_], axis=-1)
        shape = list(np.shape(k_smpl))
        la = model(np.reshape(k_smpl, (-1, 3)))
        color_index = 0
        for i in range(len(la[0])):
            volume = np.reshape(la[:,i], shape[:-1]) - fermi_energy
            if np.any(volume > 0) and np.any(volume < 0):
                cs = plt.contour(x.reshape((N, N)), y.reshape((N, N)), volume.reshape((N, N)), (0,), colors=f"C{color_index}")
                #plt.clabel(cs, inline=1, fontsize=10) # for the case where more than 1 line is plotted
                for c in cs.collections:
                    c.set_label(f"Band {i}")
                #print(i)
                color_index += 1
        plt.gca().set_aspect("equal")
        plt.title("(001)-plane cut")
        #plt.legend(loc="lower right") # doesn't work... plots really weird lines instead of just colors
        if show:
            show()