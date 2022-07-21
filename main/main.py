# Importing necessary libraries and functions
import pyvista as pv
from main.data.data import data_dictionary
from backend import calculation
from main.data import visualize, visualize2, visualize3, visualize4, visualize_all
import numpy as np
from numba import float64

class Fit(object):
    def __init__(self, points, all, n_layer): #, data, data_points_x, data_points_y
        self.data = data_dictionary()
        self.points = points # structured list, a sublist in a list - into different rows
        self.data['all_'] = all # unstructured list - one line of list acceptable by pyvista
        self.n_layer = n_layer

    def visualize_interactive(self, a, b, degree=3):
        # Opens an interactive Matplotlib CANVAS window to interact and update control points.
        if self.n_layer == 1:
            fitted_line = visualize.interactive_plotting(self, degree, a, b)
        elif self.n_layer == 2:
            fitted_line = visualize2.interactive_plotting(self, degree, a, b)
        elif self.n_layer == 3:
            fitted_line = visualize3.interactive_plotting(self, degree, a, b)
        elif self.n_layer == 4:
            fitted_line = visualize4.interactive_plotting(self, degree, a, b)
        else:
            fitted_line = visualize_all.interactive_plotting(self, degree, a, b)
        return fitted_line

    def subdivide(self, degree=3):

        # Performs subdivision on the data points with the B spline basis function
        subdivide = Fit(self.points, self.data['all_'], self.n_layer)
        self.data['degree'] = degree

        knots_list_u = []
        knots_list_v = []
        ctr_points = []
        ctr_points_plot = []
        dimension = []

        for i in range(len(self.points)):
            params_list_u, knots_list_u_ = calculation.parameterization(self.points[i], degree, u=True)
            params_list_v, knots_list_v_ = calculation.parameterization(self.points[i], degree, u=False)
            basis_matrix_u = calculation.spline_matrix(params_list_u, knots_list_u_)
            basis_matrix_v = calculation.spline_matrix(params_list_v, knots_list_v_)
            ctr_points_, ctr_points_plot_ = calculation.all_ctrl(basis_matrix_u, basis_matrix_v, self.points[i])
            dimension_ = len(ctr_points_[0])
            # ---------  append ------------- #
            knots_list_u.append(knots_list_u_)
            knots_list_v.append(knots_list_v_)
            ctr_points.append(ctr_points_)
            ctr_points_plot.append(ctr_points_plot_)
            dimension.append(dimension_)
        self.data['knots_list_u'] = knots_list_u
        self.data['knots_list_v'] = knots_list_v
        self.data['ctr_points'] = ctr_points
        self.data['ctr_points_plot'] = ctr_points_plot
        self.data['dimension'] = dimension

        subdivide.data = self.data

        return subdivide


    def datapoint_model(self):
        data_points = pv.MultiBlock()
        for i in range(len(self.data['all_'])):
            datapoint_model = visualize.create_model2(self.data['all_'][i])
            data_points.append(datapoint_model)
        return data_points

    def g(self, a, b, ctr_points_plot):
        control = pv.MultiBlock()
        for i in range(len(ctr_points_plot)):
            k = self.subsurfmodel(a, b, ctr_points_plot[i], self.data['dimension'][i], self.data['ctr_points'][i], self.data['knots_list_u'][i], self.data['knots_list_v'][i])
            control.append(k)
        return control


    def subsurfmodel(self, a, b, ctr_points_plot, dimension, ctr_points, knots_list_u, knots_list_v, degree=3):
        l = dimension
        control = [ctr_points_plot[n:n + l] for n in range(0, len(ctr_points_plot), l)]
        u = len(ctr_points)

        ctr_points_x = []
        ctr_points_y = []
        ctr_points_z = []

        for i in range(u):
            x = [am[0] for am in control[i]]
            y = [am[1] for am in control[i]]
            z = [am[2] for am in control[i]]
            ctr_points_x.append(x)
            ctr_points_y.append(y)
            ctr_points_z.append(z)

        smooth_u = [i / a for i in list(range(0, a, 1))]

        control_lenght = len(control[0])
        smooth_lenght_u = len(smooth_u)

        surface = [np.zeros((control_lenght, smooth_lenght_u)).tolist(),
                   np.zeros((control_lenght, smooth_lenght_u)).tolist(),
                   np.zeros((control_lenght, smooth_lenght_u)).tolist()]
        knots_list_u = float64(knots_list_u)

        for i in range(control_lenght):
            for_x = [x[i] for x in ctr_points_x]
            for_y = [y[i] for y in ctr_points_y]
            for_z = [z[i] for z in ctr_points_z]

            for_x = float64(for_x)
            for_y = float64(for_y)
            for_z = float64(for_z)

            surface[0][i] = [calculation.b_spline(m, knots_list_u, for_x, degree) for m in smooth_u]
            surface[1][i] = [calculation.b_spline(n, knots_list_u, for_y, degree) for n in smooth_u]
            surface[2][i] = [calculation.b_spline(o, knots_list_u, for_z, degree) for o in smooth_u]

        smooth_v = [i / b for i in list(range(0, b, 1))]  # NO longer cube but rectangle

        smooth_lenght_v = len(smooth_v)
        surface_final = np.array([np.zeros((smooth_lenght_v, smooth_lenght_u)).tolist(),
                                  np.zeros((smooth_lenght_v, smooth_lenght_u)).tolist(),
                                  np.zeros((smooth_lenght_v, smooth_lenght_u)).tolist()])

        knots_list_v = float64(knots_list_v)
        for i in range(smooth_lenght_u):
            for_x = np.array(surface[0])[:, i]
            for_y = np.array(surface[1])[:, i]
            for_z = np.array(surface[2])[:, i]

            for_x = float64(for_x)
            for_y = float64(for_y)
            for_z = float64(for_z)

            surface_final[0][:, i] = [calculation.b_spline(m, knots_list_v, for_x, degree) for m in smooth_v]
            surface_final[1][:, i] = [calculation.b_spline(n, knots_list_v, for_y, degree) for n in smooth_v]
            surface_final[2][:, i] = [calculation.b_spline(o, knots_list_v, for_z, degree) for o in smooth_v]

        i = surface_final[0]
        j = surface_final[1]
        k = surface_final[2]

        x = [item for all_sublist in i for item in all_sublist]
        y = [item for all_sublist in j for item in all_sublist]
        z = [item for all_sublist in k for item in all_sublist]
        points = np.c_[x, y, z]

        mesh = visualize.create_model(points)
        mesh.dimensions = [a, b, 1]
        return mesh

    def volumetric_mesh(self, density, layer=1):
        div = np.linspace(0, 1, density)
        if layer == 1:
            # print(len(self.data['surf_1']))
            a = self.data['surf_1_']
            b = self.data['surf_2_']

            interp_z = div * self.data['surf_2'][2][..., None] + (1 - div) * self.data['surf_1'][2][..., None]
            interp_y = div * self.data['surf_2'][1][..., None] + (1 - div) * self.data['surf_1'][1][..., None]
            interp_x = div * self.data['surf_2'][0][..., None] + (1 - div) * self.data['surf_1'][0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

        elif layer == 2:
            a = self.data['surf_2_']
            b = self.data['surf_3_']

            interp_z = div * self.data['surf_3'][2][..., None] + (1 - div) * self.data['surf_2'][2][..., None]
            interp_y = div * self.data['surf_3'][1][..., None] + (1 - div) * self.data['surf_2'][1][..., None]
            interp_x = div * self.data['surf_3'][0][..., None] + (1 - div) * self.data['surf_2'][0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

        elif layer == 3:
            a = self.data['surf_3_']
            b = self.data['surf_4_']

            interp_z = div * self.data['surf_4'][2][..., None] + (1 - div) * self.data['surf_3'][2][..., None]
            interp_y = div * self.data['surf_4'][1][..., None] + (1 - div) * self.data['surf_3'][1][..., None]
            interp_x = div * self.data['surf_4'][0][..., None] + (1 - div) * self.data['surf_3'][0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

        else:
            raise ValueError("Not enough layers!")

        pv.set_plot_theme('document')
        plotter = pv.Plotter()
        # plotter.add_mesh(b, show_scalar_bar=False)
        # plotter.add_mesh(a, show_scalar_bar=False)
        plotter.add_mesh(mesh, style='wireframe')
        plotter.show()






