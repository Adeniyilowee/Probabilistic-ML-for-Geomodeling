import pyvista as pv
import pyvista.core.pointset
import time
def interactive_plotting(all_data, degree, a, b):
    # 1.
    pl = pv.Plotter()
    all_data.subdivide(degree)
    data_points_mesh = all_data.datapoint_model()
    start = time.time()
    model_subdivided_mesh, all_data.data['points_1'] = all_data.subsurfmodel(a, b, all_data.data['ctr_points_plot'][0], all_data.data['dimension'][0], all_data.data['ctr_points'][0], all_data.data['knots_list_u'][0], all_data.data['knots_list_v'][0])
    end = time.time()
    print(end - start)

    point_ids = []
    selection = []
    selection_edge_idx = []
    radius_start = [0.1]


    # 2.1 Pyvista display setup
    def enable_sphere_widget(check):
        def change_sphere_radius(value):
            pl.clear_sphere_widgets()
            radius_start[0] = value

            pl.add_sphere_widget(update_control_cage, center=all_data.data['ctr_points_plot'][0], radius=value, color='lime', test_callback=False)


        if check:

            selection.clear()
            point_ids.clear()
            selection_edge_idx.clear()

            pl.add_slider_widget(change_sphere_radius, [0, radius_start[0] * 2], pointa=(.9, .01), pointb=(.9, .19))

        else:
            pl.clear_slider_widgets()
            pl.clear_sphere_widgets()

    # 2.1.2 updating control cage when its points are moved/manipulated
    def update_control_cage(point, idx):
        all_data.data['ctr_points_plot'][0][idx] = point
        dat, all_data.data['points_1'] = all_data.subsurfmodel(a, b, all_data.data['ctr_points_plot'][0], all_data.data['dimension'][0], all_data.data['ctr_points'][0], all_data.data['knots_list_u'][0], all_data.data['knots_list_v'][0])
        pl.add_mesh(dat, scalars=dat.points[:, 2], use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface1')




    # 3
    pl.add_checkbox_button_widget(enable_sphere_widget, size=30, color_on='lime', position=((0.3 + 0) * pl.window_size[0], 10))
    pl.add_mesh(data_points_mesh, color='aqua', point_size=4.0, render_points_as_spheres=True, name='data_points')
    pl.add_mesh(model_subdivided_mesh, scalars=model_subdivided_mesh.points[:, 2], use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface1') # , scalars=model_subdivided_mesh[:].points[:, 2]
    pl.set_background('#424242')
    pl.isometric_view_interactive()
    pl.show_axes()
    pl.show()



def create_model(vertices):
    """
    Creates pyvista.PolyData object, which can be used for plotting.

    Parameters
    ----------
    faces: (n, 4) int
          Indexes of vertices making up the faces
    vertices: (n, 3) float
          Points in space

    Returns
    ---------
    pyvista.PolyData
        pyvista object
    """
    see = pv.StructuredGrid()
    see.points = vertices
    return see


def create_model2(vertices):
    """
    Creates pyvista.PolyData object, which can be used for plotting.

    Parameters
    ----------
    faces: (n, 4) int
          Indexes of vertices making up the faces
    vertices: (n, 3) float
          Points in space

    Returns
    ---------
    pyvista.PolyData
        pyvista object
    """
    see = pv.PolyData(vertices)
    return see

def mesh_points(mesh, dim):
    layer_points_x = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -3]
    layer_points_y = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -2]
    layer_points_z = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -1]
    return [layer_points_x, layer_points_y, layer_points_z]