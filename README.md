
# PyC_Bspline_Opt
An optimized open-source python package for surfaces remapping and geological modeling

# Overview


# Quick Start

```python
from main.data import file
```
```python
# This is a work in progress!!
# Follow this ReadMe file or run the main_ex.py file
```
```python
# !!

```

```python
# Faces are defined by the indices of the vertices and have the shape (n,3) for triangular meshes.
# n is the number if faces. Each face is defined by three vertices


surfaces = [
        [[[0.0, 2, 0], [3, 2, -2], [6, 2, -5], [7, 2, -8], [9, 2, -10], [15, 2, -14]],
        [[0.0, 5, 0], [3, 5, -3], [6, 5, -5], [7, 5, -9], [9, 5, -12], [15, 5, -15]],
        [[0.0, 10, 0], [3, 10, -2], [6, 10, -5], [7, 10, -8], [9, 10, -11], [15, 10, -16]],
        [[0.0, 15, -1.0], [3, 15, -4], [6, 15, -6], [7, 15, -8], [9, 15, -11.5], [15, 15, -15]],
        [[0.0, 20, 1.0], [3, 20, -2], [6, 20, -4], [7, 20, -8], [9, 20, -11], [15, 20, -16]]],

        [[[0.0, 2, 3], [3, 2, 1], [6, 2, -2], [7, 2, -5], [9, 2, -7], [15, 2, -11]],
        [[0.0, 5, 3], [3, 5, 0], [6, 5, -2], [7, 5, -6], [9, 5, -9], [15, 5, -12]],
        [[0.0, 10, 3], [3, 10, 1], [6, 10, -2], [7, 10, -5], [9, 10, -8], [15, 10, -13]],
        [[0.0, 15, 2.0], [3, 15, -1], [6, 15, -3], [7, 15, -5], [9, 15, -8.5], [15, 15, -12]],
        [[0.0, 20, 4.0], [3, 20, 1], [6, 20, -1], [7, 20, -5], [9, 20, -8], [15, 20, -13]]],

        [[[0.0, 2, 6], [3, 2, 4], [6, 2, 1], [7, 2, -2], [9, 2, -4], [15, 2, -8]],
         [[0.0, 5, 6], [3, 5, 3], [6, 5, 1], [7, 5, -3], [9, 5, -6], [15, 5, -9]],
         [[0.0, 10, 6], [3, 10, 4], [6, 10, 1], [7, 10, -2], [9, 10, -5], [15, 10, -10]],
         [[0.0, 15, 5.0], [3, 15, 2], [6, 15, 0], [7, 15, -2], [9, 15, -5.5], [15, 15, -9]],
         [[0.0, 20, 7.0], [3, 20, 4], [6, 20, 2], [7, 20, -2], [9, 20, -5], [15, 20, -10]]],

        [[[0.0, 2, 9], [3, 2, 7], [6, 2, 4], [7, 2, 1], [9, 2, -1], [15, 2, -5]],
         [[0.0, 5, 9], [3, 5, 6], [6, 5, 4], [7, 5, 0], [9, 5, -3], [15, 5, -6]],
         [[0.0, 10, 9], [3, 10, 7], [6, 10, 4], [7, 10, 1], [9, 10, -2], [15, 10, -7]],
         [[0.0, 15, 8.0], [3, 15, 5], [6, 15, 3], [7, 15, 1], [9, 15, -2.5], [15, 15, -6]],
         [[0.0, 20, 10.0], [3, 20, 7], [6, 20, 5], [7, 20, 1], [9, 20, -2], [15, 20, -7]]]
         ]
```
```python
w = 1
data = file.read_data(surfaces, w)

subdivision_data = data.visualize_interactive(400, 400)
volumetrics = data.volumetric_mesh(10, 3)
```
# Requirements

pyvista~=0.33.0
QtPy~=2.0.0
PyQt5~=5.15.6
pyvistaqt~=0.6.0
vtk~=9.1.0
numpy~=1.22.0
setuptools~=57.0.0

# Developers
- Full stack developer : Mosaku Adeniyi (Adeniyilowee)
- Project manager and backend developer : s.Mohammad Moulaeifard (MohammadCGRE)
- Project supervisor from RWTH Aachen university: Prof. Florian Wellmann