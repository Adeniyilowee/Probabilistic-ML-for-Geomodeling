# Importing necessary libraries and functions
from scipy.stats import norm
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib as mpl
from SimPEG.utils import plot2Ddata
from IPython.core.pylabtools import figsize
from backend.geophysics_data import geophysics_data, geophysics_data_, geophysics_data_final
from backend.simulation_MCMC2 import Bayesian_Inference_Analysis as bay_inf
import pyvista as pv
from main.data.data import data_dictionary
from backend import calculation
from main.data import visualize, visualize2, visualize3, visualize4, visualize_all
from backend.simulation import probability, information_entropy, fuzziness
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


    def subsurfmodel(self, a, b, ctr_points_plot, dimension, knots_list_u, knots_list_v, degree=3):

        control = [ctr_points_plot[n:n + dimension] for n in range(0, len(ctr_points_plot), dimension)]
        u = len(control)

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
        return mesh, points

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

            # estimating the thickness to be used during MCMC as a constraint
            surf_upper = []
            surf_lower = []
            for i in range(len(np.array(self.data['surf_1']))):
                u_ = self.data['surf_1'][i].flatten()
                l_ = self.data['surf_2'][i].flatten()
                surf_upper.append(u_)
                surf_lower.append(l_)

            top = list(zip(*surf_upper))
            bottom = list(zip(*surf_lower))

            thick = []
            for i in range(len(top)):
                thickness = calculation.euc_dist(top[i], bottom[i])
                thick.append(thickness)
            self.data['thickness'] = thick
            # self.data['thickness_average'] = np.mean(thick)

        elif layer == 2:
            a = self.data['surf_2_']
            b = self.data['surf_3_']

            interp_z = div * self.data['surf_3'][2][..., None] + (1 - div) * self.data['surf_2'][2][..., None]
            interp_y = div * self.data['surf_3'][1][..., None] + (1 - div) * self.data['surf_2'][1][..., None]
            interp_x = div * self.data['surf_3'][0][..., None] + (1 - div) * self.data['surf_2'][0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

            # ---------------------- thickness -------------------------#
            # estimating the thickness to be used during MCMC as a constraint
            surf_upper = []
            surf_lower = []
            for i in range(len(np.array(self.data['surf_2']))):
                u_ = self.data['surf_2'][i].flatten()
                l_ = self.data['surf_3'][i].flatten()
                surf_upper.append(u_)
                surf_lower.append(l_)

            top = list(zip(*surf_upper))
            bottom = list(zip(*surf_lower))

            thick = []
            for i in range(len(top)):
                thickness = calculation.euc_dist(top[i], bottom[i])
                thick.append(thickness)
            self.data['thickness'] = thick
            # self.data['thickness_average'] = np.mean(thick)

        elif layer == 3:
            a = self.data['surf_3_']
            b = self.data['surf_4_']

            interp_z = div * self.data['surf_4'][2][..., None] + (1 - div) * self.data['surf_3'][2][..., None]
            interp_y = div * self.data['surf_4'][1][..., None] + (1 - div) * self.data['surf_3'][1][..., None]
            interp_x = div * self.data['surf_4'][0][..., None] + (1 - div) * self.data['surf_3'][0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

            # estimating the thickness to be used during MCMC as a constraint
            surf_upper = []
            surf_lower = []
            for i in range(len(np.array(self.data['surf_2']))):
                u_ = self.data['surf_3'][i].flatten()
                l_ = self.data['surf_4'][i].flatten()
                surf_upper.append(u_)
                surf_lower.append(l_)

            top = list(zip(*surf_upper))
            bottom = list(zip(*surf_lower))

            thick = []
            for i in range(len(top)):
                thickness = calculation.euc_dist(top[i], bottom[i])
                thick.append(thickness)
            self.data['thickness'] = thick
            # self.data['thickness_average'] = np.mean(thick)

        else:
            raise ValueError("Not enough layers!")

        self.data['mesh'] = mesh

        pv.set_plot_theme('document')
        plotter = pv.Plotter()
        # plotter.add_mesh(b, show_scalar_bar=False)
        # plotter.add_mesh(a, show_scalar_bar=False)
        plotter.add_mesh(mesh, style='wireframe')
        plotter.show()

    def flat_up(self, x_mesh, y_mesh, z_mesh):
        # this is to create dataframe that allows for index and xyz coordinates merging
        data = [(a2, b2, c2,) for a, b, c in zip(x_mesh, y_mesh, z_mesh) for a1, b1, c1 in zip(a, b, c) for a2, b2, c2 in zip(a1, b1, c1)]
        x = []
        y = []
        z = []
        for i in data:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])

        return x, y, z


    def voxel_vol(self, mesh):

        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        extent = [x_min, x_max, y_min, y_max, z_min, z_max]
        #extent = [x_min, x_max, y_min, y_max, -50, 0] # anticline
        #extent = [x_min, 100, y_min, y_max, -50, 0] # recumbent
        #extent = [x_min, x_max, y_min, y_max, -50, 0] # pinch_out



        resolution = [100, 2, 50]
        x, y, z  = self.mesh_grid(extent, resolution)
        self.x_, self.y_,self.z_ = x, y, z
        # Create unstructured grid from the structured grid
        grid = pv.StructuredGrid(x, y, z)
        uns_grid = pv.UnstructuredGrid(grid)
        # get part of the mesh within the mesh's bounding surface.

        selection = uns_grid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)
        mask_ = selection.point_data['SelectedPoints'].view(np.bool_)
        self.data['mask_bool'] = mask_
        mask = list(map(int, mask_)) # mask is thus the lith_block
        mask = list(np.float_(mask))
        #self.data['mask_dimension'] = (self.z_.shape + self.y_.shape + self.x_.shape)
        self.data['mask_dimension'] = tuple([resolution[2], resolution[1], resolution[0]]) #for simpeg
        #self.data['mask_dimension'] = tuple(resolution) # for normal geo
        mask_ = np.array(mask).reshape(self.data['mask_dimension']) # restructured mask
        #print(mask_[0].shape)
        mask = np.array(mask)
        #print(mask.shape)
        return mask, mask_



    def vol_voxelization_2(self):

        # I need this to create dataframe and extract it for gravity
        # brought it here to plot because i need to return and save mask, mask_ from voxel_vol
        density = 2.34
        mask, mask_ = self.voxel_vol(self.data['mesh'])
        self.data['lith_block'] = mask
        self.data['lith_slice'] = mask_
        self.data['mask_dataframe'] = pd.DataFrame({'orig_block': mask})
        self.data['mask_dataframe']['Density'] = self.data['mask_dataframe']['orig_block'] * density
        self.data['mask_dataframe']['X'], self.data['mask_dataframe']['Y'], self.data['mask_dataframe']['Z'] = self.flat_up(self.x_, self.y_, self.z_)

        '''
        # plotting part 1
        pv.plot(grid.points, scalars=mask)

        # plotting part 2a
        p = pv.Plotter(notebook=False)
        p.add_mesh(mask)
        p.show()
        '''
        # plotting type 2b
        fig = plt.gcf()
        im = plt.imshow(mask_[:, 1, :], origin='lower', cmap='viridis')  #
        cb = plt.colorbar(im)
        # plt.grid(None)
        plt.ylabel(r"Depth [m]")
        plt.xlabel(r"Horizontal axis [x]")
        cb.set_label(r"Lithology", rotation=270, labelpad=12, size=12)

        fig.set_size_inches(14.0, 7.0)
        fig.savefig('lith.svg', format='svg', dpi=1200)
        # plt.savefig('lithology.png', dpi=300)
        plt.show()

    def mesh_points(self, mesh, dim):
        layer_points_x = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -3]
        layer_points_y = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -2]
        layer_points_z = mesh.points.reshape(dim[:-1] + (3,), order='F')[..., -1]
        return [layer_points_x, layer_points_y, layer_points_z]

    def set_coord(self, extent, resolution):
        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]

        self.x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0],
                             dtype="float64")
        self.y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1],
                             dtype="float64")
        self.z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2],
                             dtype="float64")

        return self.x, self.y, self.z

    def create_regular_grid_3d(self, extent, resolution):

        coords = self.set_coord(extent, resolution)
        x, y, z = np.meshgrid(*coords, indexing="ij")
        return x, y, z

    def get_dx_dy_dz(self, rescale=False):
        if rescale is True:
            dx = (self.extent_r[1] - self.extent_r[0]) / self.resolution[0]
            dy = (self.extent_r[3] - self.extent_r[2]) / self.resolution[1]
            dz = (self.extent_r[5] - self.extent_r[4]) / self.resolution[2]
        else:
            dx = (self.extent[1] - self.extent[0]) / self.resolution[0]
            dy = (self.extent[3] - self.extent[2]) / self.resolution[1]
            dz = (self.extent[5] - self.extent[4]) / self.resolution[2]
        return dx, dy, dz

    def mesh_grid(self, extent, resolution):

        self.extent = np.asarray(extent, dtype='float64')
        self.resolution = np.asarray(resolution)
        self.data['resolution'] = np.asarray(resolution)
        x, y, z = self.create_regular_grid_3d(extent, resolution)
        self.dx, self.dy, self.dz = self.get_dx_dy_dz()
        return x, y, z

    def visualize_geo(self):
        final_ = self.geophysics_data(self.data['lith_slice'])
        self.data['final_geo'] = final_.flatten()
        im_vertical_slice = plt.imshow(final_[:, 1, 4:-4], origin='lower', cmap='viridis')
        cb = plt.colorbar(im_vertical_slice)
        cb.set_label(r"Gravity $(mGal)$", rotation=270, labelpad=12, size=12)
        plt.ylabel(r"Depth [m]")
        plt.xlabel(r"Horizontal axis [m]")
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        fig.savefig('geo.svg', format='svg', dpi=1200)
        # fig.savefig('geo.png', dpi=300)
        # plt.grid(None)
        plt.show()

    def visualize_geo_(self):
        points_of_interest = np.array(self.data['mesh'].points)
        resolution = self.data['mask_dimension']
        recovered_model, plotting_map, mesh = geophysics_data_(points_of_interest, resolution)
        self.data['final_geo'] = recovered_model
        dim = tuple([50, 2, 100])
        mask_ = np.array(recovered_model).reshape(dim)
        im = plt.imshow(mask_[:, 0, :], origin='lower', cmap='viridis')
        cb = plt.colorbar(im)
        cb.set_label(r"Gravity $(mGal)$", rotation=270, labelpad=12, size=12)
        plt.ylabel(r"Depth [m]")
        plt.xlabel(r"Horizontal axis [m]")
        # plt.grid(None)
        # norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
        # plt.grid(None)
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        fig.savefig('geo_pinch2.svg', format='svg', dpi=1200)
        plt.show()

    def visualize_geo_final(self):
        points_of_interest = np.array(self.data['mesh'].points)
        rec, dpred = geophysics_data_final(points_of_interest)
        self.data['final_geo'] = dpred
        # Plot
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
        plot2Ddata(rec, dpred, ax=ax1, contourOpts={"cmap": "bwr"})
        ax1.set_title("Gravity Anomaly (Z-component)")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")

        ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.85])
        norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dpred)), vmax=np.max(np.abs(dpred)))
        cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e")
        cbar.set_label("$mgal$", rotation=270, labelpad=15, size=12)
        fig.savefig('geo_pinch2.svg', format='svg', dpi=1200)
        plt.show()

    def geophysics_data(self, lith_slice):

        final_geo = []
        # but for visualization, we need to reoder it back and then use it as likelihood
        # resolution = self.data['mask_dimension']
        resolution = self.data['resolution']  # simpeg grid version
        for i in range(resolution[1]):
            slice_ = lith_slice[:, i, :]
            # print(slice_)
            geo = self.run_geo(slice_, resolution)
            final_geo.append(geo)

        # print(len(np.array(final_geo)))
        final_ = self.reshape_(final_geo)
        return final_

    def run_geo(self, slice_, resolution):

        resolution = [resolution[0], resolution[2]]
        points = []
        target = 1
        for i in range(len(slice_)):
            if target in slice_[i]:
                x_index__ = np.where(np.array(slice_[i]) == target)[0]
                x_index_ = x_index__ / resolution[0]
                y_index = i / resolution[1]

                for x_index in x_index_:
                    k = np.r_[x_index, y_index]
                    points.append(k)
        # print(points)
        f_points = np.vstack(tuple(points))
        geo = geophysics_data(f_points, resolution)
        return geo

    def reshape_(self, final_geo):

        f_geo = []
        for i in range(len(final_geo)):
            result = final_geo[i].tolist()
            f_geo.append(result)

        final_ = []
        for j in range(len(f_geo[0])):
            result_ = list(map(lambda x: x[j], f_geo))
            final_.append(result_)

        final_geo = np.array(final_)

        return final_geo


    def gaussian_MC(self, a, b, density, size):
        # this is to extract z values from a gaussian/normal distribution and use in the simulation to generate multiple grids but for MC
        dim = (a, b, 1)
        trit_top = self.distribution_1(self.data['ctr_points_plot'][0], size)
        trit_bottom = self.distribution_1(self.data['ctr_points_plot'][1], size)

        control_top = [list(x) for x in zip(*trit_top)]  # new control_list_plot
        control_bottom = [list(x) for x in zip(*trit_bottom)]

        # final_lith_block = []
        final_lith_block = np.array([])
        for i in range(len(control_top)):
            surf_top_, _ = self.subsurfmodel(a, b, control_top[i], self.data['dimension'][0], self.data['knots_list_u'][0], self.data['knots_list_v'][0], degree=3)
            surf_bottom_, _ = self.subsurfmodel(a, b, control_bottom[i], self.data['dimension'][1], self.data['knots_list_u'][1], self.data['knots_list_v'][1], degree=3)
            surf_top = self.mesh_points(surf_top_, dim)
            surf_bottom = self.mesh_points(surf_bottom_, dim)

            # -------- volume --------#
            div = np.linspace(0, 1, density)
            interp_z = div * surf_bottom[2][..., None] + (1 - div) * surf_top[2][..., None]
            interp_y = div * surf_bottom[1][..., None] + (1 - div) * surf_top[1][..., None]
            interp_x = div * surf_bottom[0][..., None] + (1 - div) * surf_top[0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

            # ---------- voxelization-----------#
            mask, mask_ = self.voxel_vol(mesh)  # they are one dimensional

            # final_lith_block.append(mask)
            final_lith_block = np.append(final_lith_block, mask)

        self.data['all_lith_block'] = final_lith_block.reshape(size, -1)
        return self.data['all_lith_block']


    def gaussian_MC_prior(self, a, b, density, size):

        # this is to extract z values from a gaussian/normal distribution and use in the simulation to generate multiple grids but for MC

        self.data['priors_0_z'] = [item[-1] for item in self.data['ctr_points_plot'][0]]  # original
        self.data['priors_1_z'] = [item[-1] for item in self.data['ctr_points_plot'][1]]  # original

        # Model to test our algorithm, introducing completely new Z values
        #
        cov_0 = np.diag([2.5] * len(self.data['priors_0_z']))
        self.data['Prior_z_top'] = np.random.multivariate_normal(self.data['priors_0_z'], cov_0, 1).T.flatten()
        cov_1 = np.diag([2.5] * len(self.data['priors_1_z']))
        self.data['Prior_z_bottom'] = np.random.multivariate_normal(self.data['priors_1_z'], cov_1, 1).T.flatten()

        prior_t = self.data['ctr_points_plot'][0]
        prior_b = self.data['ctr_points_plot'][1]

        for i in range(len(prior_t)):
            prior_t[i][-1] = self.data['Prior_z_top'][i]

        for i in range(len(prior_b)):
            prior_b[i][-1] = self.data['Prior_z_bottom'][i]

        dim = (a, b, 1)
        trit_top = self.distribution_1(prior_t, size)
        trit_bottom = self.distribution_1(prior_b, size)

        control_top = [list(x) for x in zip(*trit_top)]  # new control_list_plot
        control_bottom = [list(x) for x in zip(*trit_bottom)]

        # final_lith_block = []
        final_lith_block = np.array([])
        for i in range(len(control_top)):
            surf_top_, _ = self.subsurfmodel(a, b, control_top[i], self.data['dimension'][0], self.data['knots_list_u'][0], self.data['knots_list_v'][0], degree=3)
            surf_bottom_, _ = self.subsurfmodel(a, b, control_bottom[i], self.data['dimension'][1], self.data['knots_list_u'][1], self.data['knots_list_v'][1], degree=3)
            surf_top = self.mesh_points(surf_top_, dim)
            surf_bottom = self.mesh_points(surf_bottom_, dim)

            # -------- volume --------#
            div = np.linspace(0, 1, density)
            interp_z = div * surf_bottom[2][..., None] + (1 - div) * surf_top[2][..., None]
            interp_y = div * surf_bottom[1][..., None] + (1 - div) * surf_top[1][..., None]
            interp_x = div * surf_bottom[0][..., None] + (1 - div) * surf_top[0][..., None]
            volume = interp_z[..., None] * [0, 0, 1] + interp_y[..., None] * [0, 1, 0] + interp_x[..., None] * [1, 0, 0]
            mesh = pv.StructuredGrid(*volume.transpose())

            # ---------- voxelization-----------#
            mask, mask_ = self.voxel_vol(mesh)  # they are one dimensional

            # final_lith_block.append(mask)
            final_lith_block = np.append(final_lith_block, mask)

        self.data['all_lith_block'] = final_lith_block.reshape(size, -1)
        return self.data['all_lith_block']

    def gaussian_MCMC(self, density):
        density = density
        self.simulation = bay_inf(self, self.a, self.b, density)

        set_likelihood = self.simulation.likelihood(self.data['final_geo'])

        shape_0 = len(self.data['priors_0_z'])
        sigma_0 = 5
        shape_1 = len(self.data['priors_1_z'])
        sigma_1 = 5

        set_top_control = self.simulation.top_prior(self.data['priors_0_z'], shape_0, sigma_0)
        set_bottom_control = self.simulation.bottom_prior(self.data['priors_1_z'], shape_1, sigma_1)

        number_of_steps = 25000
        burnin = 1000
        num_leapfrog_steps = 3
        step_size = 0.7  # or
        # step_size = tf.compat.v1.get_variable(name='step_size' , initializer=tf.constant(0.5, dtype=tf.float32)

        # setting up the method
        method = 'HMC'  #'RWM'

        initial_chain_state = [tf.constant(self.data['Prior_z_top'], name='top_init', dtype=tf.float64), tf.constant(self.data['Prior_z_bottom'], name='bottom_init', dtype=tf.float64)]  # self.data['priors_1_z']

        # Initialize any created variables.
        all_traces = self.simulation.tf_MCMC(method, number_of_steps, burnin, step_size, num_leapfrog_steps, initial_chain_state)
        print('Trace_Done, Nice!')

        return all_traces

    def MAP_model(self):
        # Model to test our algorithm, introducing completely new Z values
        # But for now, since we are comparing HMC with MAP, otherwise we would have created it (see above)
        priors_0_z_unc = self.data['Prior_z_top']
        priors_1_z_unc = self.data['Prior_z_bottom']

        # Methods = ['Sgd', 'Nadam', 'Adag', 'Adam']
        top_list, bottom_list, self.data['cost_A'], self.data['cost_B'] = self.simulation.MAP_estimate('Adam', priors_0_z_unc, priors_1_z_unc)

        if len(top_list) < 100:
            final_lith_block = np.array([])
            size = len(top_list)
            for i in range(size):
                mask, mask_, mesh = self.simulation.uncertainty(top_list[i], bottom_list[i])

                final_lith_block = np.append(final_lith_block, mask)

            self.data['map_lith_block'] = final_lith_block.reshape(size, -1)


        else:
            final_lith_block = np.array([])
            top_list_ = top_list[-100:]
            bottom_list_ = bottom_list[-100:]
            size = len(top_list_)
            for i in range(size):
                mask, mask_, mesh = self.simulation.uncertainty(top_list_[i], bottom_list_[i])

                final_lith_block = np.append(final_lith_block, mask)

            self.data['map_lith_block'] = final_lith_block.reshape(size, -1)

        # init_g = tf.compat.v1.global_variables_initializer()
        # init_l = tf.compat.v1.local_variables_initializer()
        # evaluate(init_g)
        # evaluate(init_l)

    def plot_posterior_distribution(self):

        top_hist = pd.DataFrame(self.data['top_posterior_']).add_prefix('control_top')
        bottom_hist = pd.DataFrame(self.data['bottom_posterior_']).add_prefix('control_bottom')

        top_hist['control_top0'].hist(edgecolor='yellow', linewidth=1.2, figsize=(90, 90), bins=40)
        bottom_hist['control_bottom0'].hist(edgecolor='yellow', linewidth=1.2, figsize=(90, 90), bins=40)
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        plt.show()

    def plot_traces(self):
        lw = 1
        x = np.arange(0, len(self.data['top_posterior_'][:, 0]))
        # for i in range(len(self.data['top_posterior_'][0])):
        plt.plot(x, self.data['top_posterior_'][:, 0], label="new trace of center 0", lw=lw, c="#5DA5DA")
        plt.show()

    def error(self):
        df = pd.DataFrame(self.data['cost_B'], columns=['Loss'])  #
        plt.plot(df.index, df["Loss"], label="Adam Optimizer")  # [0::100]
        plt.ylabel(r"Loss")
        plt.xlabel(r"Iteration")
        plt.legend(loc='upper right')
        plt.title('Learning Rate: 0.001')
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        #fig.savefig('error.svg', format='svg', dpi=1200)
        plt.show()

    def error_(self):
        df = pd.DataFrame(self.data['cost_A'], columns=['Loss'])  # [1:]
        plt.plot(df.index, df["Loss"], label="Adam Optimizer")  # [0::100]
        plt.ylabel(r"Loss")
        plt.xlabel(r"Iteration")
        plt.legend(loc='upper right')
        plt.title('Learning Rate: 0.001')
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        #fig.savefig('test2png.png', dpi=100)
        plt.show()

    def prior_top(self):

        top_prior_dist = np.random.normal(self.data['Prior_z_top'][0], 1, size=len(np.array(self.data['top_posterior_'][:, 0])))
        top_hist_df = pd.DataFrame(top_prior_dist, columns=['Prior'])

        mu_prior = top_hist_df['Prior'].mean()
        std_prior = top_hist_df['Prior'].std()

        top_hist_df.plot.hist(bins=50, alpha=.7, color="sandybrown", density=True)
        mn, mx = plt.xlim()
        mm, my = plt.ylim()
        x_prior = np.linspace(mn, mx, 200)
        p_prior = norm.pdf(x_prior, mu_prior, std_prior)
        plt.plot(x_prior, p_prior, color="darkred", linewidth=2)
        plt.vlines(top_prior_dist.mean(), ymin=mm, ymax=my, label="Prior score mean", linestyles="-.", color="darkred")

        plt.xlim(mn, mx)
        plt.legend(loc="upper right");
        plt.xlabel("Control Point [m]")
        plt.ylabel("Probability")
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        #fig.savefig('prior_top.svg', format='svg', dpi=1200)
        plt.show()

    def prior_posterior_top(self):

        top_posterior_dist = np.array(self.data['top_posterior_'][:, 0])
        top_prior_dist = np.random.normal(self.data['Prior_z_top'][0], 1, size=len(top_posterior_dist))
        top_hist_df = pd.DataFrame(list(zip(top_posterior_dist, top_prior_dist)), columns=['Posterior', 'Prior'])

        mu_prior = top_hist_df['Prior'].mean()
        std_prior = top_hist_df['Prior'].std()
        mu_posterior = top_hist_df['Posterior'].mean()
        std_posterior = top_hist_df['Posterior'].std()

        top_hist_df.plot.hist(bins=50, alpha=0.7, density=True)

        mn, mx = plt.xlim()
        mm, my = plt.ylim()

        x_prior = np.linspace(mn, mx, 200)
        x_posterior = np.linspace(mn, mx, 200)

        p_prior = norm.pdf(x_prior, mu_prior, std_prior)
        p_posterior = norm.pdf(x_posterior, mu_posterior, std_posterior)

        plt.plot(x_prior, p_prior, color="brown", linewidth=2)
        plt.plot(x_posterior, p_posterior, color="royalblue", linewidth=2)

        plt.vlines(top_prior_dist.mean(), ymin=mm, ymax=my, label="Prior score mean", linestyles="-.", color="brown")
        plt.vlines(top_posterior_dist.mean(), ymin=mm, ymax=my, label="Posterior score mean", linestyles="-.", color="royalblue")

        plt.xlim(mn, mx)
        plt.ylim(mm, my)
        plt.legend(loc="upper right");
        plt.xlabel("Control Point [m]")
        plt.ylabel("Probability")
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        #fig.savefig('prior_posterior.svg', format='svg', dpi=1200)
        plt.show()



    def simulation(self):
        # this is simply for the evaluation of voxels
        # This probability block contains the probability of all the geology block in the grid
        # Hence, if we have block [0, 1, 2], we will be able to view them individually with respect to others using the index
        self.data['prob_block'] = probability(self.data['all_lith_block'])
        # Entropy basically brings out the areas of all the geology blocks in the grid based on their probability,
        # thus where we have 1, is black for all and between 1 and 0 for areas that are not sure
        self.data['entropy_block'] = information_entropy(self.data['prob_block'])
        # self.data['fuzziness'] = fuzziness(self.data['prob_block'])

    def map_simulation(self):
        # this is simply for the evaluation of voxels
        # This probability block contains the probability of all the geology block in the grid
        # Hence, if we have block [0, 1, 2], we will be able to view them individually with respect to others using the index
        self.data['prob_block'] = probability(self.data['map_lith_block'])
        # Entropy basically brings out the areas of all the geology blocks in the grid based on their probability,
        # thus where we have 1, is black for all and between 1 and 0 for areas that are not sure
        self.data['entropy_block'] = information_entropy(self.data['prob_block'])
        # self.data['fuzziness'] = fuzziness(self.data['prob_block'])

    def probability(self):
        # Plot
        prob_block = np.array(self.data['prob_block'][1]).reshape(self.data['mask_dimension'])
        extent = [self.extent[0], self.extent[1], self.extent[2], self.extent[3]]

        im = plt.imshow(prob_block[:, 1, :], origin='lower', cmap='viridis')  # , extent = extent
        # plt.grid(None)
        cb = plt.colorbar(im)
        cb.set_label(r"Probability $P(\theta)$", rotation=270, labelpad=12, size=12)
        plt.ylabel(r"Depth [m]")
        plt.xlabel(r"Horizontal axis [m]")
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        #fig.savefig('probabilty.svg', format='svg', dpi=1200)
        plt.show()

    def entropy(self):

        # Plot
        entropy_block = np.array(self.data['entropy_block']).reshape(self.data['mask_dimension'])
        extent = [self.extent[0], self.extent[1], self.extent[2], self.extent[3]]

        im = plt.imshow(entropy_block[:, 1, :], origin='lower', cmap='viridis')  # , extent = extent
        cb = plt.colorbar(im)
        # plt.grid(None)
        plt.ylabel(r"Depth [m]")
        plt.xlabel(r"Horizontal axis [m]")
        cb.set_label(r"Entropy $(H)$", rotation=270, labelpad=12, size=12)
        fig = plt.gcf()
        fig.set_size_inches(14.0, 7.0)
        fig.savefig('entropy.svg', format='svg', dpi=1200)
        # fig.savefig('entropy.png', dpi=300)
        plt.show()

        '''    
        def fuzziness_MC(self):
        # Plot
        fuzziness = np.array(self.data['fuzziness']).reshape(self.data['mask_dimension'])
        extentt = [self.extent[0], self.extent[1], self.extent[2], self.extent[3]]

        im = plt.imshow(fuzziness[:, 1, :], cmap='viridis', extent = extentt)
        plt.colorbar(im)
        plt.show()
        '''
    def distribution(self, data, size):
        np.random.seed(1234)
        #mean = data[:, -1].flatten().mean()
        zt = []
        for i in range(len(data)):
            u = data[i]
            mean = u[-1]
            distribution = np.random.normal(mean, 5, size=size)
            litr = []
            for j in range(len(distribution)):
                z = u.copy()  # wow, saved my life!
                z[-1] = distribution[j]
                litr.append(z)
            zt.append(litr)
        return zt

    def distribution_1(self, data, size):
        np.random.seed(1234)
        #mean = data[:, -1].flatten().mean()
        zt = []
        for i in range(len(data)):
            u = data[i]
            distribution = np.random.normal(0, 1, size=size)
            litr = []
            for j in range(len(distribution)):
                z = u.copy()  # wow, saved my life!
                z[-1] = distribution[j]+z[-1]
                litr.append(z)
            zt.append(litr)
        return zt

