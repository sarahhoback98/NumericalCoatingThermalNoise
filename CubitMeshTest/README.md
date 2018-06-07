This code modifies the dealii step-4 test at
https://www.dealii.org/8.4.0/doxygen/deal.II/step_4.html

Instead of solving the Laplace equation on a cube, 
as in the original test, this code reads in a mesh from a mesh-generator 
and solves the Laplace equation. You must supply the mesh and edit the 
file name and type of grid in Step4<dim>::make_grid().

