ABOUT THIS REPO
=====================

This respository holds C++ code that numerically computes Brownian coating and substrate thermal noise for amorphic and crystalline materials. Results from this code are presented in https://doi.org/10.1088/1361-6382/aa9ccc.

If you make use of this code, please cite technical report https://doi.org/10.1088/1361-6382/aa9ccc.

The primary requirement for compiling and running this code is the deal.II framework (http://www.dealii.org). It also requires petsc, and you'll need MPI to run on multiple processors.

BUILDING AND USING
=====================

Instructions for building this code, and example use cases, will be added here.

Meanwhile, once you have installed deal.ii (dealii.org) and petsc (petsc.org) with the hypre package and p4est (and other dealii dependences), you can build and run the code "QuasistaticBrownianThermalNoise.cpp" similarly to how you build and run the deal-ii step-8 tutorial (https://www.dealii.org/8.4.0/doxygen/deal.II/step_8.html).

I recommend first ensuring that you can build and run the deal.II tutorials step-3 (Laplace equation), step-8 (2D elastic equations), and step-17 (petsc + MPI Laplace equation).

You'll need to install the following dependencies:
  * [yaml-cpp](https://github.com/jbeder/yaml-cpp)
  
To build yaml-cpp, you can do this:
```
git clone https://github.com/jbeder/yaml-cpp
cd yaml-cpp
git checkout yaml-cpp-0.6.2
mkdir -p ~/Codes/yaml/0.6.2
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Codes/yaml/0.6.2 ..
make
ctest
make install
```

To build and run the thermal noise code, e.g., do this:

```
cd /path/to/QuasistaticBrownianThermalNoise.cpp
mkdir build
cd build

cmake -DYAMLCPP_ROOT=/path/to/yamlcpp ..

make release #or just "make"...if a debug version of dealii is 
             #installed, make release avoids it
```

Then, after editing `config.yaml`, run the code with, e.g.,

```
mpirun -np 12 QuasistaticBrownianThermalNoise.cpp #run on 12 cores
```

To run on a cluster, you might use a batch submission script; an example 
script is provided.

CHOOSING THE PHYSICS
=====================
Currently, you must edit the source code to change most of the physics 
(e.g., different material, different mirror dimensions, etc.) However, these settings 
will move to the input file `config.yaml.`

The relevant parameters are set in ElasticProblem::ElasticProblem().

Starting on line 465, you can choose the following:

  - mTKOutput = save vtk data for making 3D images (e.g. via paraview)?
  - mNumberOfCycles = numer of resolutions ("cycles", ~13 is a good choice)
  - mWhichCoatingYijkl = coating material (choose from those coded)
  - mWhichSubstrateYijkl = substrate material (choose from those coded)
  - r0 = beam width
  - F0 = amplidude of applied pressure (should not affect results)
  - rad = radius of cylindrical mirror
  - d = coating thickness
  - halflength = half of cylinder length along axis
  
Additionally, you might wish to change the temperature from 300K, which is set
on line 1321 in a variable called T. (Search for T = 300) to find the line.

ADDING YOUR OWN MATERIALS
=========================

You can add your own materials. Steps involved:
1. Choose a name for your material (say NAME), and add kNAME to the enumerate 
just under the last #include
2. Add Tensor<4,dim> Y_NAME to ElasticProblem's class definition.
3. Define Y_NAME in the ElasticProblem constructor. (i.e. Yijkl)
4. Define lossPhi_NAME in the ElasticProblem constructor (i.e. the loss angle).


QuasistaticBrownianThermalNoise_Paper.cpp
============================================

For completeness, this is the version of the code used in https://dcc.ligo.org/LIGO-P1700183. This version is not as polished but gives equivalent results.

Testing
============================================
A sample output is ExampleEnergy.dat. Build and run QuasistaticBrownianThermalNoise.cpp without 
changing any code, and this should be your output.

QUESTIONS
===========================================
Please send questions to glovelace at fullerton dot edu.
