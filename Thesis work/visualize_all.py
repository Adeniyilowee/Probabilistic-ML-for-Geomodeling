import pyvista as pv
import time
def interactive_plotting(all_data, degree, a, b):
    # 1.
    pl = pv.Plotter()

    all_data.subdivide(degree)
    ctr_points_plot = all_data.data['ctr_points_plot'].copy()
    dim = len(ctr_points_plot[0])
    #et = len(ctr_points_plot)
    val = [item for all_sublist in ctr_points_plot for item in all_sublist]
    data_points_mesh = all_data.datapoint_model()
    start = time.time()
    model_subdivided_mesh = all_data.g(a, b, ctr_points_plot)
    end = time.time()
    print(end - start)
    point_ids = []
    selection = []
    selection_edge_idx = []
    radius_start = [0.1]


    # 2. Pyvista display setup
    def enable_sphere_widget(check):
        def change_sphere_radius(value):
            pl.clear_sphere_widgets()
            radius_start[0] = value

            pl.add_sphere_widget(update_control_cage, center=val, radius=value, color='lime', test_callback=False)

        if check:

            selection.clear()
            point_ids.clear()
            selection_edge_idx.clear()

            pl.add_slider_widget(change_sphere_radius, [0, radius_start[0] * 2], pointa=(.9, .01), pointb=(.9, .19))

        else:
            pl.clear_slider_widgets()
            pl.clear_sphere_widgets()

    # 3. updating control cage when its points are moved/manipulated
    def update_control_cage(point, idx):
        val[idx] = point
        ctr_points_plot = [val[n:n + dim] for n in range(0, len(val), dim)]
        pl.add_mesh(all_data.g(a, b, ctr_points_plot), use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface') # scalars=all_data.g(a, b, ctr_points_plot)[:].points[:, 2],

    pl.add_checkbox_button_widget(enable_sphere_widget, color_on='red', position=((0.3 + 0) * pl.window_size[0], 10))
    pl.add_mesh(data_points_mesh, color='aqua', point_size=4.0, render_points_as_spheres=True, name='data_points')
    pl.add_mesh(model_subdivided_mesh, use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface')
    pl.set_background('#424242')
    pl.isometric_view_interactive()
    pl.show_axes()
    pl.show()



def create_model(vertices):

    see = pv.StructuredGrid()
    see.points = vertices
    return see


def create_model2(vertices):

    see = pv.PolyData(vertices)
    return see# .delaunay_2d()


def mesh_points(mesh, dim):
    layer_points = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -1]
    layer_grid_2d = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., :-1]
    return [layer_points, layer_grid_2d]