This respository holds C++ code that numerically computes Brownian coating and substrate thermal noise for amorphic and crystalline materials. Results from this code are presented in https://dcc.ligo.org/LIGO-P1700183.

The primary requirement for compiling and running this code is the deal.II framework (http://www.dealii.org). It also requires petsc, and you'll need MPI to run on multiple processors.

Instructions for building this code, and example use cases, will be added here. Meanwhile, once you have installed deal.ii (dealii.org) and petsc (petsc.org) with the hypre package, you can build and run the code "QuasistaticBrownianThermalNoise.cpp" similarly to how you build and run the deal-ii step-8 tutorial (https://www.dealii.org/8.4.0/doxygen/deal.II/step_8.html).

I recommend first ensuring that you can build and run the deal.II tutorials step-3 (Laplace equation), step-8 (2D elastic equations), and step-17 (petsc + MPI Laplace equation).

To run the code, e.g., do this:

cd /path/to/QuasistaticBrownianThermalNoise.cpp
cmake .
make release #or just "make"...if a debug version of dealii is 
             #installed, make release avoids it
mpirun -np 12 QuasistaticBrownianThermalNoise.cpp

To run on a cluster, you might use a batch submission script; an example 
script is provided.

Please send questions to glovelace at fullerton dot edu.
