import pyvista as pv
import time


def interactive_plotting(all_data, degree, a, b):
    # 1.
    pl = pv.Plotter()
    all_data.subdivide(degree)
    data_points_mesh = all_data.datapoint_model()
    start = time.time()
    dim = (a, b, 1)
    model_subdivided_mesh, all_data.data['points_1'] = all_data.subsurfmodel(a, b, all_data.data['ctr_points_plot'][0], all_data.data['dimension'][0], all_data.data['knots_list_u'][0], all_data.data['knots_list_v'][0])
    model_subdivided_mesh_1, all_data.data['points_2'] = all_data.subsurfmodel(a, b, all_data.data['ctr_points_plot'][1], all_data.data['dimension'][1], all_data.data['knots_list_u'][1], all_data.data['knots_list_v'][1])
    all_data.data['surf_1_'] = model_subdivided_mesh
    all_data.data['surf_2_'] = model_subdivided_mesh_1
    all_data.data['surf_1'] = mesh_points(model_subdivided_mesh, dim)
    all_data.data['surf_2'] = mesh_points(model_subdivided_mesh_1, dim)
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
        dat, all_data.data['points_1'] = all_data.subsurfmodel(a, b, all_data.data['ctr_points_plot'][0], all_data.data['dimension'][0], all_data.data['knots_list_u'][0], all_data.data['knots_list_v'][0])
        all_data.data['surf_1'] = mesh_points(dat, dim)
        all_data.data['surf_1_'] = dat
        pl.add_mesh(dat, color='sandybrown', use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface1')


    # 2.2 Pyvista display setup 2
    def enable_sphere_widget1(check):
        def change_sphere_radius(value):
            pl.clear_sphere_widgets()
            radius_start[0] = value
            pl.add_sphere_widget(update_control_cage1, center=all_data.data['ctr_points_plot'][1], radius=value, color='blue', test_callback=False)


        if check:

            pl.add_slider_widget(change_sphere_radius, [0, radius_start[0] * 2], pointa=(.9, .01), pointb=(.9, .19))

        else:
            pl.clear_slider_widgets()
            pl.clear_sphere_widgets()

    # 2.2.2. updating control cage when its points are moved/manipulated
    def update_control_cage1(point, idx):
        all_data.data['ctr_points_plot'][1][idx] = point
        dat, all_data.data['points_2'] = all_data.subsurfmodel(a, b, all_data.data['ctr_points_plot'][1], all_data.data['dimension'][1], all_data.data['knots_list_u'][1], all_data.data['knots_list_v'][1])
        all_data.data['surf_2'] = mesh_points(dat, dim)
        all_data.data['surf_2_'] = dat
        pl.add_mesh(dat, color='darkgray', use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface2')


    # 3
    pl.add_checkbox_button_widget(enable_sphere_widget, size=30, color_on='lime', position=((0.3 + 0) * pl.window_size[0], 10))
    pl.add_checkbox_button_widget(enable_sphere_widget1, size=30, color_on='blue', position=((0.35 + 0) * pl.window_size[0], 10))
    pl.add_mesh(data_points_mesh, color='aqua', point_size=4.0, render_points_as_spheres=True, name='data_points')
    pl.add_mesh(model_subdivided_mesh, color='sandybrown', use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface1')
    pl.add_mesh(model_subdivided_mesh_1, color='darkgray', use_transparency=False, show_edges=False, pickable=False, name='subdivided_surface2')  # scalars=model_subdivided_mesh_1.points[:, 2]
    #
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
    return see


def mesh_points(mesh, dim):
    layer_points_x = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -3]
    layer_points_y = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -2]
    layer_points_z = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -1]
    return [layer_points_x, layer_points_y, layer_points_z]
