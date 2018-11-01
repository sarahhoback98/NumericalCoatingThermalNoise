/* ---------------------------------------------------------------------
 *
 * Elastostatic mirror by Geoffrey Lovelace.
 *
 * This code uses deal.II, which is (C) 2000 - 2015 by the deal.II authors
 * This file is based on the step-8 tutorial.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Geoffrey Lovelace, Cal State Fullerton, 2017
 * https://arxiv.org/abs/1707.07774
 * https://dcc.ligo.org/LIGO-P1700183
 *
 * This code has been tested with deal.ii v8.2.1.
 */

#include <deal.II/base/quadrature_lib.h> //
#include <deal.II/base/function.h> //
#include <deal.II/base/timer.h> //

// Support parallelization with petsc + p4est
#include <deal.II/lac/generic_linear_algebra.h>
#define USE_PETSC_LA

namespace LA
{
#ifdef USE_PETSC_LA
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <deal.II/lac/vector.h> //
#include <deal.II/lac/full_matrix.h> //
#include <deal.II/lac/solver_cg.h> //
#include <deal.II/lac/constraint_matrix.h> //
//#include <deal.II/lac/compressed_simple_sparsity_pattern.h> //

#include <deal.II/lac/petsc_parallel_sparse_matrix.h> //
#include <deal.II/lac/petsc_parallel_vector.h> //
#include <deal.II/lac/petsc_solver.h> //
#include <deal.II/lac/petsc_precondition.h> //

#include <deal.II/grid/grid_generator.h> //
#include <deal.II/grid/tria_accessor.h> //
#include <deal.II/grid/tria_iterator.h> //
#include <deal.II/dofs/dof_handler.h> //
#include <deal.II/dofs/dof_accessor.h> //
#include <deal.II/dofs/dof_tools.h> //
#include <deal.II/fe/fe_values.h> //
#include <deal.II/fe/fe_system.h> //** Support the vector-valued problem
#include <deal.II/fe/fe_q.h> // Q1 finite elements (linear tent shape funcs)

#include <deal.II/numerics/vector_tools.h> //
#include <deal.II/numerics/data_out.h> //
#include <deal.II/numerics/error_estimator.h> //

// For cylinder domain
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/base/utilities.h> //
#include <deal.II/base/conditional_ostream.h> //
#include <deal.II/base/index_set.h> // // determine which cells are owned or 
                                       // known about on this proc
#include <deal.II/lac/sparsity_tools.h> //

// Distributed meshes and functions to refine them
#include <deal.II/distributed/tria.h> //
#include <deal.II/distributed/grid_refinement.h> //

#include <fstream> //
#include <iostream> //

#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/logstream.h> 
#include <iomanip>
#include <cmath>

//Support for reading in grids in potential future versions
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <typeinfo> //get type of pointer, used in looping over cells

//Support for yaml reading a configuration file
#include <yaml-cpp/yaml.h>

//****************************************************************//  
// Enumerate different available Yijkl
//****************************************************************//  
enum material { kAlGaAs, kIso_AlGaAs, kIso_Ta2O5, kIso_FusedSilica };

enum profiles { TEM00, TEM02, TEM20, TEM02minus20 };

//****************************************************************//  
// Date and Time Function         
//****************************************************************//   
// Return the current date and time as an array of characters
// Printed to assist in profiling
const std::string currentDateTime() {
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

  return buf;
}

namespace QuasistaticBrownianThermalNoise
{
  using namespace dealii;

//****************************************************************//  
// Cylinder translation function
//****************************************************************//   
//The dealii primitive cylinder has an axis on the x axis.
//This rotates and shifts the cylinder so the cylinder domain has 
//an axis parallel to the z axis, with the coating-substrate
//boundary at z=0.

struct CylTransFunc
{
  //Define the CylTransFunc struct ("public class")
  CylTransFunc(double coatThick, double halfCylThick);
  double d,halflength;
  Point<3> operator() (const Point<3> &in) const
  {
    return Point<3> (in(2),
                     in(1),
		     -1.*in(0)-(halflength-d));
    //subtract d in above, not d./2: halfCylThick is 
    //r/2 + d/2, and to end with coating of 
    //thickness d, then I want to drop by r/2 - d/2
    //I.e. if I drop by halfThick, the top of the coating
    //is at z=0. If I drop by halfthick-d, I get a coating 
    //thickness d
  }
};

//Constructor just initializes member variables with passed in 
//parameters.
CylTransFunc::CylTransFunc(double coatThick, double halfCylThick):
  d(coatThick),
  halflength(halfCylThick)
{  
}


  //****************************************************************//
  //ElasticProblem interface
  //****************************************************************//

  //This class handles setting up and solving the problem

  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem (); //sets constants innerMirrorSize,outFac,r0,a
    ~ElasticProblem ();
    void run ();

  private:
    void parse_config_file();
    void setup_system ();
    void assemble_system ();
    unsigned int solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle);    

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    DoFHandler<dim>      dof_handler;
    FESystem<dim>        fe;    

    // Determine which cells are locally owned, and which are locally relevant
    // (e.g. owned + ghosts)
    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix     constraints;

    //Do not define a SparsityPattern, as petsc has its own
    //SparsityPattern      sparsity_pattern; // petsc has its own sparsity 
                                             // handler
    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;
    


    ConditionalOStream pcout; //Output stream that only prints on proc0

    // Beam profile
    int beam_profile;

    //Built-in Young's tensors
    //Y_Iso_FusedSilica = fused silica
    //Y_AlGaAs = AlGaAs x=0.92 crystal
    //Y_Iso_AlGaAs = effective isotropic x=0.92 AlGaAs
    //Y_Iso_Ta2O5 = Ta2O5 isotropic coating material
    //Young tensors Yijkl
    Tensor<4,dim>        Y_Iso_FusedSilica, Y_AlGaAs, 
      Y_Iso_Ta2O5, Y_Iso_AlGaAs;

    //The built-in Young's tensors have lame_mu and lame_lambda defined
    //These are the Lame parameters used to construct Y
    double               lame_mu_FusedSilica, lame_lambda_FusedSilica, 
      lame_mu_Ta2O5, lame_lambda_Ta2O5, 
      lame_mu_AlGaAs, lame_lambda_AlGaAs;

    //Mirror dimensions
    double r0, F0, d;

    //Loss angles for each built-in material
    // lossPhi_Iso_FusedSilica = fused silica
    // lossPhi_Iso_Ta2O5 = Tantalum oxide
    // lossAlGaAsx92 = x=0.92 AlGaAs
    // lossPhiAlGasIso = loss angle when using effective isotropic AlGaAs
    double lossPhi_Iso_FusedSilica, lossPhi_Iso_Ta2O5, lossPhi_AlGaAs, lossPhi_Iso_AlGaAs;

    //Mirror dimension parameters
    double               innerMirrorSize, outFac, rad, substrateheight, halflength;

    //Timing
    TimerOutput                               computing_timer;    

    //Other options
    bool mVTKOutput;
    unsigned int mNumberOfCycles;
    int mWhichCoatingYijkl;
    int mWhichSubstrateYijkl;

    double getYijkl(int whichYijkl, unsigned int i, 
		    unsigned int j, unsigned int k, unsigned int l);
  };

  //****************************************************************//
  // Neumann values interface
  //****************************************************************//

  //This class computes the Neumann (traction) boundary values

  template <int dim>
  class NeumannValues :  public Function<dim>
  {
  public:
    NeumannValues (const double& inR0, const double& inF0, const int& in_beam_profile);
    double r0,F0;
    int beam_profile;

    //Return the x,y,z values of the Neumann value at a point
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    //Return the Neumann values at a list of points
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };

  //****************************************************************//
  // Right hand side interface
  //****************************************************************//

  // Note: all vector-valued functions have to have a
  // constructor, since they need to pass down to the base class of how many
  // components the function consists. 
  template <int dim>
  class RightHandSide :  public Function<dim>
  {
  public:
    RightHandSide ();
    
    //Return x,y,z components of right hand side
    virtual void vector_value (const Point<dim> /*&p*/,
                               Vector<double>   &values) const;
    //Return the RHS at several points
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };

  //****************************************************************//
  // Neumann values implementation
  //****************************************************************//

  // The constructor passes down to the base class the number of 
  //components (dim=3).

  template <int dim>
  NeumannValues<dim>::NeumannValues (const double& inR0, const double& inF0, const int& in_beam_profile)
    :
    Function<dim> (dim),
    r0(inR0),
    F0(inF0),
    beam_profile(in_beam_profile)
  {}

  // arXiv:1707.07774 Eq. (33) third integrand, Eq. (27), etc. Tzz = -F p(r), 
  // Txz=Tyz=0
  template <int dim>
  inline
  void NeumannValues<dim>::vector_value (const Point<dim> &p,
                                         Vector<double>   &values) const
  {
    //Make sure the vector I passed in has 1 component per dimension
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    //This code assumes 3D for simplicity (will not work in 2D without changes)
    Assert (dim >= 2, ExcNotImplemented());

    //The Neumann value is \bar{T}_{nj}. Here n is the unit normal to the 
    //surface, which points in the +z direction, since the Neuman boundary
    //is at z=0, with the rest of the surface at z<0. So This is 
    //\bar{T}_{zj}. I want only a z pressure, so \bar{T}_{zx}=\bar{T}_{zy}=0/
    //\bar{T}_{zz} is set to a Gaussian centered at the origin.

    //Note: F0, r0 set in constructor

    //First, set all values to zero.
    for(unsigned int i=0; i<dim-1; ++i) {
      values(i) = 0.;
    }

    //Second, set the z value = \bar{T}_{zz} = F0 exp[(-x^2-y^2)/r0^2] in 3D
    double myValue = 0.0;
    for(unsigned int i=0; i<dim-1; ++i) {
      myValue -= p(i)*p(i);            
    }
    //std::cout << "myValue: " << myValue << std::endl;

    //The negative sign here is because Tzj should point in the -z 
    //direction, so the mirror is compressed, not stretched.
    //Normalize: e.g. Liu+ Eq. (2): if F0=F0, I'd better divide by pi*r0^2
    switch(beam_profile) {
    case(TEM00):
      myValue = -1.0 * F0 * exp(myValue/(r0*r0));
      myValue /= M_PI * r0 * r0;
      break;
    case(TEM02):
      myValue = -1.0 * F0 * exp(myValue/(r0*r0)) * (4.0*p(1)*p(1)/(r0*r0)-2.0)*(4.0*p(1)*p(1)/(r0*r0)-2.0);
      myValue /= 8.0 * M_PI * r0 * r0;
      break;
    case(TEM20):
      myValue = -1.0 * F0 * exp(myValue/(r0*r0)) * (4.0*p(0)*p(0)/(r0*r0)-2.0)*(4.0*p(0)*p(0)/(r0*r0)-2.0);
      myValue /= 8.0 * M_PI * r0 * r0;
      break;
    case(TEM02minus20):
      myValue = -1.0 * F0 * exp(myValue/(r0*r0)) * ((4.0*p(1)*p(1)/(r0*r0)-2.0)*(4.0*p(1)*p(1)/(r0*r0)-2.0)-(4.0*p(0)*p(0)/(r0*r0)-2.0)*(4.0*p(0)*p(0)/(r0*r0)-2.0));
      myValue /= 8.0 * M_PI * r0 * r0;
      break;
    default:
      std::cout << "WARNING! Invalid profile string. Do not trust results!\n";
      break;
    }

    //Set the Neuman value: values(0) = T_zx, values(1) = T_zy, values(2)=Tzz
    //dim=3, so values(dim-1) = Tzz.
    //Note: this assumes only the top of the cylinder has a pressure applied
    //Dirichlet conditions elsewhere.
	  
    values(dim-1) = myValue;
    
  }

  // Return the Neumann value at several points. 
  template <int dim>
  void NeumannValues<dim>::vector_value_list 
  (const std::vector<Point<dim> > &points,
   std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      NeumannValues<dim>::vector_value (points[p],
                                        value_list[p]);
  }

  //****************************************************************//
  // Right hand side implementation
  //****************************************************************//

  template <int dim>
  RightHandSide<dim>::RightHandSide ()
    :
    Function<dim> (dim)
  {}

  template <int dim>
  inline
  void RightHandSide<dim>::vector_value (const Point<dim> /*&p*/,
                                         Vector<double>   &values) const
  {
    //DealII convention: check return value size immediately
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));
    Assert (dim >= 2, ExcNotImplemented());

    //Set the right-hand side to zero. Aside from the Neumann condition
    //to be implemented elsewhere, the system is in equilibrium.
    //This is the RHS of -\nabla_i T_{ij} = 0. (cf. arXiv:1707.07774 Eq. (25)
    //and second integrand in arXiv:1707.07774 Eq. (33))
    for(unsigned int i=0; i<dim; ++i) {
      values(i) = 0.;
    }
  }

  //Return value at many points, calling the single-point function each time
  //(so you only have to change RHS code in one place)
  template <int dim>
  void RightHandSide<dim>::vector_value_list 
  (const std::vector<Point<dim> > &points,
   std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();


    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value (points[p],
                                        value_list[p]);
  }

  //****************************************************************//
  // ElasticProblem implementation
  //****************************************************************//
  template <int dim> double
  ElasticProblem<dim>::getYijkl(int whichYijkl, unsigned int i, 
				unsigned int j, unsigned int k, 
				unsigned int l) {
    //Choices are 
    //kAlGaAs #crystal, x=0.92
    //kIso_AlGaAs #effective isotropic approx to x=0.92 AlGaAs
    //kIso_Ta2O5 #tantalum
    //kIso_FusedSilica #fused silica
    switch(whichYijkl) {
    case(kAlGaAs): 
      return Y_AlGaAs[i][j][k][l];
    case(kIso_AlGaAs): 
      return Y_Iso_AlGaAs[i][j][k][l];
    case(kIso_Ta2O5):
      return Y_Iso_Ta2O5[i][j][k][l];
    case(kIso_FusedSilica):
      return Y_Iso_FusedSilica[i][j][k][l];
    default:
      std::cout << "WARNING! Invalid Yijkl string. Do not trust results!\n";
      break;
    }
    return 0.0;
  }

  template <int dim>
  void ElasticProblem<dim>::parse_config_file() {
    YAML::Node config = YAML::LoadFile("config.yaml");

    const std::string beam_profile_string = config["BeamProfile"].as<std::string>();
    if (beam_profile_string == "TEM00") {
        beam_profile = TEM00;
    } else if (beam_profile_string == "TEM02") {
        beam_profile = TEM02;
    } else if (beam_profile_string == "TEM20") {
        beam_profile = TEM20;
    } else if (beam_profile_string == "TEM02minus20") {
        beam_profile = TEM02minus20;
    } else {
        std::cout << "ERROR: Invalid BeamProfile in config.yaml\n\n";
        exit(-1);
    }

    std::cout << "Beam profile: " << beam_profile
              << " (" << config["BeamProfile"].as<std::string>() << ")\n";
    exit(0);
  }

  template <int dim>
  ElasticProblem<dim>::ElasticProblem ()
    :
    mpi_communicator (MPI_COMM_WORLD), //MPI_COMM_WORLD=all procs available
    //keep triangulation smooth
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    fe (FE_Q<dim>(2), dim), //use 3D quad elements, and use dim=3
                            //elements to make a vector element from 3
                            //scalar elements.  
                            //use 3D lin elements with 2->1 here 
                            //and QGauss=2 instead of QGauss=3 later on
                            //for faster speed, but quad seems to be
                            //more accurate, at least for smooth solutions.
    pcout (std::cout,
	   (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    Y_Iso_FusedSilica(),
    Y_AlGaAs(),
    Y_Iso_Ta2O5(),
    Y_Iso_AlGaAs(),
    //Initialize a timer
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times),
    mVTKOutput(false),
    mNumberOfCycles(3) // resolutions to do: 0,1,...mNumberOfCycles - 1

  {
    // Parse input file
    parse_config_file();


    //Choose which Yijkl to use for substrate and which to use for coating
    //Choices are 
    //kAlGaAs #crystal, x=0.92
    //kIso_AlGaAs #effective isotropic approx to x=0.92 AlGaAs
    //kIso_Ta2O5 #tantalum
    //kIso_FusedSilica #fused silica
    mWhichCoatingYijkl = kAlGaAs;
    mWhichSubstrateYijkl = kIso_FusedSilica;

    // I hereby declare code units so that 1 = 1 TPa for stress tensor
    //                                     1 = beam size for length

    // Intialize beam width w (in same units as innerMirrorSize) 
    // and beam amplitude a (=F_0, in code units)
    r0 = 176.77669534; //note: our "r0" is r_0 in Liu & Thorne (2000)
                       //cf. arXiv:1707.07774 Eq. (4)   
                       // 176.7 from Cole+ (2013)
    F0 = 0.001; //final result should not depend on this


    // Choose the size of the mirror: it's a rectangle with points
    // low = (-innerMirrorSize*outFac, -innerMirrorSize*outFac, -innerMirrorSize*outFac)
    // up  = (innerMirrorSize*outFac, innerMirrorSize*outFac, 0)
    // Later, I will insert a coating above z=0, and move the outer boundary.
    rad = 25000/2.0; //25800.; 
    innerMirrorSize = 4.*r0; // this isn't the full size, but is used to help
                       // concentrate points near the spacer
    outFac = rad / innerMirrorSize; //so total mirror size is rad
                               //diameter of substrate is 25000 micron in 
                               //Cole+ (2013).

    d = 6.83; //coating extends above z=0 this far...
              //6.83 micron: Cole+ 2013
              //4.68 based on Table I of Chalermsongsak+ (2015)
              //earlier versions of this code used 4.68
      
    substrateheight = 25000.0/2.0;


    //halflength = rad/8.+d/2.; //0.25 in + coating = total thickness
    halflength = substrateheight/2.+d/2.; //1.0 in + coating = total thickness

    
    ////////////////////////////////////////////////////////
    //nothing below should need changing for different runs
    ////////////////////////////////////////////////////////

    // For isotropic materials, choose the Lame parameters.
    // Here: Fused Silica (amorphous SiO2)
    // Units: 1 = 1 TPa = 1e12 Pa
    // e.g. Y=0.073, sigma=0.17...fused silica (E=73 GPa, nu=0.17, 
    
    //Cole+ 2013 uses Y=72 Gpa, sigma=0.17 for substrate, 
    //Y=100 GPa, sigma=0.32 coating
    //Y = Young's modulus, sigma = Poisson Ratio, 
    //K = bulk modulus, mu = shear modulus
    //Y = 9 mu K / (3K + mu), sigma = (3K - 2 mu)/(2(3k+mu)), K=lambda+(2/3)mu

    lame_lambda_FusedSilica = 0.0158508;
    lame_mu_FusedSilica = 0.0307692;

    //Set Y_Iso_FusedSilica   
    for(int i=0;i<dim;++i) {
      for(int j=0;j<dim;++j) {
	for(int k=0;k<dim;++k) {
	  for(int l=0;l<dim;++l) {
	    
	    //Y_{ijkl} = lambda dij dkl + mu dik djl + mu dil djk
	    if(i==j && k==l) {
	      Y_Iso_FusedSilica[i][j][k][l] += lame_lambda_FusedSilica;
	    }
	    
	    if(i==k && j==l) {
	      Y_Iso_FusedSilica[i][j][k][l] += lame_mu_FusedSilica;
	    }
	    
	    if(i==l && j==k) {
	      Y_Iso_FusedSilica[i][j][k][l] += lame_mu_FusedSilica;
	    }
	    
	  }
	}
      }
    }  

    // For isotropic materials, choose the Lame parameters.
    // Here: Ta2O5
    // Units: 1 = 1 TPa = 1e12 Pa
    // E=0.073, nu=0.17...fused silica (E=73 GPa, nu=0.17, 
    // via accuratus.com/fused.html)
    // wikipedia's numbers are similar

    lame_lambda_Ta2O5 = 0.0484794;
    lame_mu_Ta2O5     = 0.0569106;

    //Set Y_Ta2O5   
    for(int i=0;i<dim;++i) {
      for(int j=0;j<dim;++j) {
	for(int k=0;k<dim;++k) {
	  for(int l=0;l<dim;++l) {
	    
	    //Y_{ijkl} = lambda dij dkl + mu dik djl + mu dil djk
	    if(i==j && k==l) {
	      Y_Iso_Ta2O5[i][j][k][l] += lame_lambda_Ta2O5;
	    }
	    
	    if(i==k && j==l) {
	      Y_Iso_Ta2O5[i][j][k][l] += lame_mu_Ta2O5;
	    }
	    
	    if(i==l && j==k) {
	      Y_Iso_Ta2O5[i][j][k][l] += lame_mu_Ta2O5;
	    }
	    
	  }
	}
      }
    }  

    // For isotropic materials, choose the Lame parameters.
    // Here: AlGaAs effective isotropic
    // Units: 1 = 1 TPa = 1e12 Pa
    // E=0.100, nu=0.32...fused silica (E=100 GPa, nu=0.32), 
    // via accuratus.com/fused.html)
    // wikipedia's numbers are similar

    lame_lambda_AlGaAs = 0.0673401;
    lame_mu_AlGaAs     = 0.0378788;

    //Set Y_Iso_AlGaAs   
    for(int i=0;i<dim;++i) {
      for(int j=0;j<dim;++j) {
	for(int k=0;k<dim;++k) {
	  for(int l=0;l<dim;++l) {
	    
	    //Y_{ijkl} = lambda dij dkl + mu dik djl + mu dil djk
	    if(i==j && k==l) {
	      Y_Iso_AlGaAs[i][j][k][l] += lame_lambda_AlGaAs;
	    }
	    
	    if(i==k && j==l) {
	      Y_Iso_AlGaAs[i][j][k][l] += lame_mu_AlGaAs;
	    }
	    
	    if(i==l && j==k) {
	      Y_Iso_AlGaAs[i][j][k][l] += lame_mu_AlGaAs;
	    }
	    
	  }
	}
      }
    }  


    //Set Y to a cubic crystal Yijkl
    //Specifically, use Al_{x}Ga_{1-x}As

    //Set the x in Al_{x}Ga_{1-x}As, i.e., the aluminum fraction
    //Chalermsongsak+ (2015) uses x=0.92
    //double x = 0.92; //uncomment if using the Gehrsitz+ formulas.

    //Gehrsitz+, PRB 60, 11601 (1999) gives [their Eq. (2)]:
    //Units: TPa, i.e., 10^{12} Pa.
    //double c11 = 0.188; // 0.188 +/- 0.7 TPa
    //double c12 = 0.0537 + 0.00485*x + 0.0119*x*x - 0.0130*x*x*x;
    //double c44 = 0.0591-0.00188*x;

    //Just use Cole+ (2013) elastic moduli.
    //Supplemental document, section S2.
    double c11 = 119.94/1000.0;
    double c12 = 55.38/1000.0;
    double c44 = 59.15/1000.0;

    //The following code sets all 21 nonzero components. 
    //Prepared in Mathematica, then sorted here by hand. 
    //I also checked this by eye.
    Y_AlGaAs[0][0][0][0] =c11;
    Y_AlGaAs[1][1][1][1] =c11;
    Y_AlGaAs[2][2][2][2] =c11;    

    Y_AlGaAs[0][0][1][1] =c12;
    Y_AlGaAs[0][0][2][2] =c12;
    Y_AlGaAs[1][1][0][0] =c12;
    Y_AlGaAs[1][1][2][2] =c12;
    Y_AlGaAs[2][2][0][0] =c12;
    Y_AlGaAs[2][2][1][1] =c12;

    Y_AlGaAs[0][1][0][1] =c44;
    Y_AlGaAs[0][1][1][0] =c44;
    Y_AlGaAs[1][0][0][1] =c44;
    Y_AlGaAs[1][0][1][0] =c44;

    Y_AlGaAs[0][2][0][2] =c44;
    Y_AlGaAs[0][2][2][0] =c44;
    Y_AlGaAs[2][0][0][2] =c44;
    Y_AlGaAs[2][0][2][0] =c44;

    Y_AlGaAs[1][2][1][2] =c44;
    Y_AlGaAs[1][2][2][1] =c44;
    Y_AlGaAs[2][1][1][2] =c44;
    Y_AlGaAs[2][1][2][1] =c44;

    // Define some dimensionless loss angles.
    // lossPhi_Iso_FusedSilica = fused silica
    // lossPhi_Iso_Ta2O5 = Tantalum oxide
    // lossAlGaAsx92 = x=0.92 AlGaAs
    //lossPhi = 1.e-7; //Chalermsongsak+ (2015) Table I
    lossPhi_Iso_FusedSilica = 1.e-6; //Cole+ (2013)
    lossPhi_Iso_Ta2O5 = 4.e-4; //Yamamoto+ (2006), PRD 74, 022002 
                               // (good value??)
    lossPhi_Iso_AlGaAs = 2.5e-5; //Cole+ (2013)
    lossPhi_AlGaAs = lossPhi_Iso_AlGaAs; //use same loss angle from Cole+(2013)



  } //end of ElasticProblem constructor

  // The destructor: force dof handler to delete before finite element object
  // to avoid undefined pointers (see step-6 tutorial for details)
  template <int dim>
  ElasticProblem<dim>::~ElasticProblem ()
  {
    dof_handler.clear ();
  }

  template <int dim>
  void ElasticProblem<dim>::setup_system ()
  {
    TimerOutput::Scope t(computing_timer, "setup");   

    // Distribute degrees of freedom (shape functions)
    // Note: if you want to renumber to partially diagonalize, do that here
    // (don't think that's necessary)
    dof_handler.distribute_dofs (fe);

    //locally owned, locally relevant dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
					     locally_relevant_dofs);

    //Initialize the system matrix, RHS, and solution vector
    //Must give the communicator, global sizes, and local sizes
    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);

    // Hanging node note: with local mesh refinement, you might get 
    // this:
    //                    //x = original node, + = refined node
    //  x--+--x-----x     //o = hanging node
    //  |  |  |     |
    //  +--+--o     |     //The big cell's shape functions should be 
    //  |  |  |     |     //linear on the boundaries.
    //  x--+--x--o--x     //But points o are affected by the small 
    //  |  |  |  |  |     //cells, which only try to be linear betwen xo.
    //  +--+--+--+--+
    //  |  |  |  |  |     //The hanging node constraint makes sure 
    //  x--+--x--+--x     //xox is linear. See dealii constraints on DOF
    //                    //documentation for details.

    //Constraints handle Dirichlet condition and hanging-node constraints
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs); //if not locally relevant only,
                                                //could use huge amount of 
                                                //memory
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(dim),
                                              constraints);
    constraints.close ();

    //Initialize the matrix. Use CompressedSimpleSparsityPattern
    //if you don't yet know the final sparsity pattern, DynamicSparsityPattern
    //works better than
    //CompressedSimpleSparsityPattern csp (locally_relevant_dofs);
    DynamicSparsityPattern csp (locally_relevant_dofs);

    DoFTools::make_sparsity_pattern (dof_handler, csp,
                                     constraints, false);
    SparsityTools::distribute_sparsity_pattern 
      (csp, 
       dof_handler.n_locally_owned_dofs_per_processor(),
       mpi_communicator,
       locally_relevant_dofs); //ensure each proc knows all relevant dofs

    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          csp,
                          mpi_communicator);
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system ()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    //Gauss quadrature order n exact for polynomials up to order 2n-1.

    //Here shape functions are either i) linear, so gauss quad. order 2 OK
    //for 2*2-1 = cubic. Worst case is square of function, i.e. quadratic, or
    //ii) quadratic, so gauss quadrature order 3 good enough 
    //for 3*3-1 = 8, while worst case is square of function, i.e. quartic.

    // Have to use sufficiently accurate quadrature formula. In addition, we
    // need to compute integrals over faces, i.e. 2D (dim-1) objects.
    QGauss<dim>  quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);


    //set up regular values as usual
    //also set up face values -- see step 7 for details...face values
    //are for surface integrals for the neumann condition
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values | update_quadrature_points 
				      | update_normal_vectors 
				      | update_JxW_values);


    const unsigned int   dofs_per_cell = fe.dofs_per_cell;

    const unsigned int   n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    RightHandSide<dim>      right_hand_side;
    std::vector<Vector<double> > rhs_values (n_q_points,
                                             Vector<double>(dim));

    // The Neumann values are set up the same way as the right hand side,
    // except that we pass in the beam size and amplitude.
    // We don't precompute them here. Instead, we get them on the faces later.
    NeumannValues<dim>      neumann(r0,F0,beam_profile);

    // Now we can begin with the loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell!=endc; ++cell) {
      if (cell->is_locally_owned()) {
	
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);

        // Get the RHS at the quadrature points. 
        right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                                           rhs_values);

	// A double to store the value of Yijkl
	double Yijkl = 0.;

	// Assemble the matrix. A,B are the dofs in this cell. 
	// M_{AB} = Y_{ijkl} \Phi_{Aj;i} \Phi_{Bl;k}
	// In this notation, my vector shape functions are
	// \Phi_{Aj}, where A is the dof number and j specifies which 
	// component. The sum will only have one term survive: j=comp(A).
	// Similarly, in the l sum, the only surviving term is l=comp(B).
	//
	// That means that the matrix becomes
	//
	// M_{AB} = Y_{icomp(A)kcomp(B)} \Phi_{Acomp(A);i} \Phi_{Bcomp(B);k}
	// so I only must sum over i and k.
	//       
	// Eq. (38) of arXiv:1707.07774: integrand of first integral
	// Eq. (39) of arXiv:1707.07774

	//DOF loops
	for (unsigned int A=0; A<dofs_per_cell; ++A) {
	  const unsigned int
	    component_A = fe.system_to_component_index(A).first;
	  for (unsigned int B=0; B<dofs_per_cell; ++B) {
	    const unsigned int
	      component_B = fe.system_to_component_index(B).first;

	    //Vector component loops
	    for (unsigned int i=0; i<dim; ++i) {
	      for(unsigned int k=0; k<dim; ++k) {

		//Loop over quadrature points in the cell
		for (unsigned int q_point=0; q_point<n_q_points; q_point++) {

		  // First, choose Yijkl. In principle, this could depend on 
		  // the location of the quadrature point. But for now,
		  // just use a single Yijkl tensor in coat, single in sub.
		  
		  // E.g. AlGaAs coating, fused silica substrate
		  // find the quadrature point. If z > 0, it's in the coating.
		  // Otherwise, it's in the substrate.
		  if(fe_values.quadrature_point(q_point)(dim-1) > 0.) {
		    Yijkl = 
		      getYijkl(mWhichCoatingYijkl,i,component_A,
			       k,component_B);
		  } else {
		    Yijkl = 
		      getYijkl(mWhichSubstrateYijkl,i,component_A,
			       k,component_B);
		  }

		  // Now, the matrix element picks up a term 
		  // Y_{icomp(A)kcomp(B)} \phi_{Acomp(A);i} \phi_{Bcomp(B);k}
		  cell_matrix(A,B)
		    += Yijkl 
		    * fe_values.shape_grad(A,q_point)[i]
		    * fe_values.shape_grad(B,q_point)[k]
		    * fe_values.JxW(q_point);
		  
		}
		
	      }
	    }
	    
	  }
	}

	//Now assemble the right hand side
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const unsigned int
            component_i = fe.system_to_component_index(i).first;

            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              cell_rhs(i) += fe_values.shape_value(i,q_point) *
                             rhs_values[q_point](component_i) *
                             fe_values.JxW(q_point);
          }
	
	// Take care of the surface integral

	// Neumann boundary has indicator=1; Dirichlet has indicator=0.

	//loop over faces
        for (unsigned int face_number=0; 
	     face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
          if (cell->face(face_number)->at_boundary()
              &&
              (cell->face(face_number)->boundary_id() == 1))
            {
              // If we came into here, then we have found an external face
              // belonging to the Neumann condition. 

	      //re_init computes quantities we need on the face...
	      //see step-8 for details.
              fe_face_values.reinit (cell, face_number);

              // And we can then perform the integration by using a loop over
              // all quadrature points.


              for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {

		  Vector<double> neumann_value(dim);
		  neumann.vector_value 
		    (fe_face_values.quadrature_point(q_point),
		    neumann_value);


		  for (unsigned int i=0; i<dofs_per_cell; ++i) {
		    
		    const unsigned int
		      component_i = fe.system_to_component_index(i).first;

		    cell_rhs(i) += (neumann_value(component_i) *
				    fe_face_values.shape_value(i,q_point) *
				    fe_face_values.JxW(q_point));
		  }
		}
	    }
      
      

	//Insert results for this cell into the full matrix, vector, etc.
        cell->get_dof_indices (local_dof_indices);
	//Treat the constraints now, to avoid having to talk to other
	//procs later, while also distributing local to global objects.
	constraints
          .distribute_local_to_global(cell_matrix, cell_rhs,
                                      local_dof_indices,
                                      system_matrix, system_rhs);

      } //end if cell owner is this mpi proc
    } //end loop over cells

    // Now compress the vector and the system matrix:
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    //Note from Geoffrey: ZeroFunction here because we impose Dirichlet 
    //boundary conditions on boundaries flagged as zero (all but the z=0 face).
    //ZeroFunction<dim>(dim) sets all dim=3 vector components: each becomes 
    //a dim=3 scalar function that's zero everywhere.

    //See step-17 for why add the false here: save time by not forcing 
    //the system_matrix to be symmetric again...turns out not to slow down
    //the petsc solver much.

    //See step-40 for details: do not separately impose Dirichlet 
    //conditions; these are part of the constraints now.
  }

  //solver doesn't care how the matrix and vector were put together.
  //so same as 1 core, except now use petsc!
  //CG solver: works because matrix is positive definite, symmetric
  //Hypre preconditioner: scales well (see step-40 for details)
  template <int dim>
  unsigned int ElasticProblem<dim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    
    LA::MPI::Vector
    completely_distributed_solution (locally_owned_dofs, mpi_communicator);

    //Set up a SolverControl convergence monitor and then the petsc solver
    SolverControl           solver_control (dof_handler.n_dofs(), 
					    1.e-12);
    LA::SolverCG cg (solver_control, mpi_communicator);

    //hypre parasails preconditioner works well, others I tested
    //not well
    PETScWrappers::PreconditionParaSails preconditioner;
    PETScWrappers::PreconditionParaSails::AdditionalData data;
    data.symmetric = true;

    preconditioner.initialize(system_matrix, data);


    //solve the system using petsc's conjugate gradient linear solver
    cg.solve (system_matrix, completely_distributed_solution, system_rhs,
              preconditioner);

    pcout << "   Solver: solved in " << solver_control.last_step()
          << " iterations." << std::endl;

    //to deal with the constraints, get a localized copy of the solution
    //then distribute hanging node constraints on the local copy, since
    //this operation needs everything
    
    constraints.distribute (completely_distributed_solution);

    //copy everything back to the global vector, including ghosts
    locally_relevant_solution = completely_distributed_solution;

    //return number of iterations it took to converge
    return solver_control.last_step();
  }

  //Refining the grid now involves communication. 
  //1. Compute error indicators for cells on this proc
  //2. Distribute the refinement indicators so all procs know all indicators
  //3. Then copy full indicator vector so this proc knows all of them
  template <int dim>
  void ElasticProblem<dim>::refine_grid ()
  {
    TimerOutput::Scope t(computing_timer, "refine");   

    //Compute the local error
    Vector<float> local_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate 
      (dof_handler,
       QGauss<dim-1>(3),
       typename FunctionMap<dim>::type(),
       locally_relevant_solution,
       local_error_per_cell); //which subdomain? (i.e., which proc?)

    //Load up the global error vector
    //See step-17 for details...set up a parallel vector

    //for now, each proc refines separately
    //future version: refine based on single global error estimate
    //refine the grid:
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number 
      (triangulation,
       local_error_per_cell,
       0.14, 0.02);

    pcout << "Grid refinement created. Level before refinement: "
	  << triangulation.n_global_levels() << "."
	  << std::endl;


    triangulation.execute_coarsening_and_refinement ();

    pcout << "Grid refined. Level after refinement: "
	  << triangulation.n_global_levels() << "."
	  << std::endl;

  }

  template <int dim>
  void ElasticProblem<dim>::output_results (const unsigned int cycle) 
  {
    TimerOutput::Scope t(computing_timer, "output");
    //VTK output if flag enabled
    if (mVTKOutput) {

      DataOut<dim> data_out;
      data_out.attach_dof_handler (dof_handler);

      std::vector<std::string> solution_names;
      switch (dim)
	{
	case 1:
	  solution_names.push_back ("displacement");
	  break;
	case 2:
	  solution_names.push_back ("x_displacement");
	  solution_names.push_back ("y_displacement");
	  break;
	case 3:
	  solution_names.push_back ("x_displacement");
	  solution_names.push_back ("y_displacement");
	  solution_names.push_back ("z_displacement");
	  break;
	default:
	  Assert (false, ExcNotImplemented());
	}
      data_out.add_data_vector (locally_relevant_solution, solution_names);

      //color cells by which process they belong to
      //first, make a vector, 1 per cell, containing the subdomain number
      Vector<float> subdomain (triangulation.n_active_cells());
      for (unsigned int i=0; i<subdomain.size(); ++i)
	subdomain(i) = triangulation.locally_owned_subdomain();
      data_out.add_data_vector (subdomain, "subdomain");

      data_out.build_patches ();

      const std::string filename = ("solution-" +
				    Utilities::int_to_string (cycle, 2) +
				    "." +
				    Utilities::int_to_string
				    (triangulation.locally_owned_subdomain(), 
				     4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);


    //Write the master file
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 2) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");

        std::ofstream master_output ((filename + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
    }

    //*********************************************************************//
    // Post-process: compute the energy.
    //*********************************************************************//

    //first, get the gradient of the solution

    //nQuad = 1 is sufficient to get beautiful convergence and accuracy 
    //for the total energy. But resolving the cotating vs substrate is 
    //harder. I call a point in the coating or substrate based on where
    //the quadrature point is: above or below z=0. But the quadrature 
    //point placement varies with the mesh refinement; I believe this
    //is why the coating energy doesn't show strict convergnece.
    //However, the coating energy varies much less with increasing nQuad.
    //nQuad=10 dominates the run time but gives a more consistent
    //value for the coating noise. Increasing nQuad lays down more 
    //quadrature points (subdividing the cells nQuad times on each axis and 
    //doing 3rd order Gaussian quadrature in each subdivision).

    const int nQuad = 10; //how many times to iterate the quadrature
                         //only do this with grids designed to 
                         //have cells only in the coating or substrate, 
                         //as opposed to cells part in one, part in the other
    QGauss<1>  quadrature_formula_base(3);
    QIterated<dim> quadrature_formula(quadrature_formula_base, nQuad);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    //const unsigned int           dofs_per_cell = fe.dofs_per_cell; //unused
    const unsigned int           n_q_points    = quadrature_formula.size();

    std::vector<std::vector<Tensor<1, dim> > > 
      solution_gradients(n_q_points, 
			 std::vector<Tensor<1,dim> >(dim));

    //second, get the Yijkl. In this code, it is taken to be isotropic 
    //and homogeneous.
    
    //Now loop over cells, adding each cell's energy to the total
    double local_energy = 0.;
    double local_coatingEnergy = 0.;
    double local_substrateEnergy = 0.;
    double local_thisEnergy = 0.;
    typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
    for (; cell!=endc; ++cell) { 
      if (cell->is_locally_owned()) {
	fe_values.reinit(cell);


          //The integral over a cell becomes a sum over the cell's quadrature
	  //points. Now, if the cell volume is dV (including the jacobian), 
	  //then we want to add (1/2) S_{ij} T_{ij} dV to the cell energy.

	  // S_{ij} T_{ij} = S_{ij} Y_{ijkl} S_{kl}
	  //               = Y_{ijkl} \nabla_{i} u_{j} \nabla_{k} u_l
	  //               = Y_{ijkl} u_{j;i} u_{l;k}
	  //
	  // Here u_{j;i} is the gradient of vector j in the ith direction.
	  //

      //get function gradients at the quadrature points:
      //solution_gradients[q][c][d] is deriv in direction d of cth vector
      //at quadrature point q. So u_{j;i}(x_q) = solution_gradients[q][j][i]
	fe_values.get_function_gradients(locally_relevant_solution,solution_gradients);
      
	//store current quadrature point's Young's modulus
	double theYijkl = 0.;

	//Do the vector component loops
	for(unsigned int q_point=0; q_point < n_q_points; ++q_point) {
	  for(int i=0;i<dim;++i) {
	    for(int j=0;j<dim;++j) {
	      for(int k=0;k<dim;++k) {
		for(int l=0;l<dim;++l) {

		  local_thisEnergy =
		    0.5 
		    * fe_values.JxW(q_point)
		    * solution_gradients[q_point][j][i]
		    * solution_gradients[q_point][l][k];		    

		if(fe_values.quadrature_point(q_point)(dim-1) > 0.) {
		  theYijkl = getYijkl(mWhichCoatingYijkl,i,j,k,l);
		  local_thisEnergy *= theYijkl;
		  local_coatingEnergy += local_thisEnergy;
		} else {
		  theYijkl = getYijkl(mWhichSubstrateYijkl,i,j,k,l);
		  local_thisEnergy *= theYijkl;
		  local_substrateEnergy += local_thisEnergy;
		}

		// Add this quadrature point's contribution to the 
		// elastic energy.
		local_energy += local_thisEnergy;
	      }	      
	    }
	  }
	}
      }
	
      }
    }
    


    //create a vector to hold the energy values: 1 per process

    double energy = Utilities::MPI::sum(local_energy,mpi_communicator);
    double substrateEnergy = Utilities::MPI::sum(local_substrateEnergy,
						 mpi_communicator);
    double coatingEnergy =  Utilities::MPI::sum(local_coatingEnergy,
						mpi_communicator);
    //Normalize energy: Fluctuation-dissipation theorem divides out F0^2
    //dependence. I just do this here.
    energy /= F0*F0;
    substrateEnergy /= F0*F0;
    coatingEnergy /= F0*F0;

    //I now have U/F0^2 = Uo/Fo^2. The thermal noise is given by 
    //(e.g. Hong+ Eq. (46) S = (4 k_B T / pi f) (U/F0^2) \phi
    //So the code can compute f S = (4 k_B T / pi) \phi (U/F0^2)
    //I can put in effective loss angles \phi for different materials: 
    //Fused silica, Tantalum, and AlGaAs with x=0.92.
    //Note: I used 1=1TPa=1e12 Pa before, so I'm a factor of 1e12 away from SI
    //units

    //compute noise (amplitude spectral density) in SI units (1/sqrt(Hz))
    //first, multiply by 1.e-12, since I set 1=1e12 Pa previously
    //then, multiply by 4/L^2 (Blandford & Thorne (10.59)) to get S_h from 
    //S_q. Finally, take the sqrt to get the amplitude spectral density.

    //Check units (note F_o has units of Newtons)
    //
    // Hong+ Eq. (46): [S_x] = [J/K] [K] [s] [J] [N]^{-2}
    //                       = [N][m][K]^{-1}[K][s][N][m][N]^{-2}
    //                       = [N][N][N]^{-2} [K]^{-1}[K] [m][m][s]
    //                       = [m]^2 [s] 
    //                       = [m]^2 / [Hz]
    // So the sqrt of this has units of [m] / sqrt[Hz]
    // Multiply the amplitude spectral density by 2/L to get units 1/sqrt{Hz}
    // since (Blandford and Thorne Eq. (10.59)) S_h = 4/L^2 S_q.
    // Now I'm working almost in SI units, except I 
    //       i) work in units of microns, so lengths are in microns
    //      ii) Pressures are in units of 1e12 Pa.
    // Let's convert U/(F0*F0) to SI, then put everything else in SI
    //
    //  The energy density U is 
    //  u ~ Y_{ijkl} S_{ij} S_{kl}. S is dimensionless. 
    //
    // But is S really dimensionless in my screwy units? I guess I'm solving 
    // for displacement in microns. Then, I take a gradient to get a 
    // dimensionless quantity. But at the same time, S should be linear in
    // the amplitude: more displacement for more applied force.
    // Let's see: Lovelace+ (2008) Eq. (14e): S_{ij} has units of 
    // \tilde{p} / \mu * k^2, where k has units of 1/length. On the 
    // surface, S_{zz} (Eq. (15d) has the units of (1/lame) * p(r).
    // For a Gaussian, p(r) = F_0 e^{-r^2/r0^2} / (\pi r0^2).
    // So my S_{ij} really has units of (Fo / micron^2) * (1/lame).
    // Let's say Fo is in code units, or 1=1e12 N. Then my Lame coefficients
    // are in units of TPa or 1e12 N / m^2 = 1e12 J / m^3. So then
    // my strain has units of (1e12 N / micron^2) / (1e12 N / m^2)
    // = m^2 / micron^2 = m^2 / (1e-6 m)^2 = 1e12.
    // So each factor of strain picks up 1e12, so u now has units
    // [1e24] * Y.
    //
    // So U has the same units as Y but must be multiplied by 1e24.
    // Y_Iso_FusedSilica and Y_IsoTa2O5 have units of lame_lambda, lame_mu
    // Units of lame parameters: they are given in units of TPa = 1e12 N/m^2
    // = 1e12 J/m^3 * (m/1e6 micron)^3 = 1e-6 J/micron^3.
    // Then I do a volume integral, effectively turning u into an energy U,
    // which has units 1e12 J/m^3 (1e-6 m)^3 = 1e-6 J.
    // So my energy has units of 1e-6 J * 1e24 = 1e18 J. 
    // 
    // Then, I divide by Fo^2. Fo is in "code units", which means 1 = 1e12 N.
    // So I should first convert Fo into Newtons by multiplying by 1e12.
    // So U/F0^2 has units [1e18 J][1e12 N]^{-2} = 1e-6 m/N.
    // So the total power spectral density has units (N m/Hz)(N m)/(N^2) = 
    // m^2/Hz, as expected. So U/F0^2 * 1e-6 is a quantity in meters/Newton. 
    // So U/F0^2 has units 1e-6 m/N and is now in SI. Multiplty by 4kT/pi
    // in SI units to get a quantity that is in units of m^2. Divide by 
    // frequency to get m^2/Hz. Take sqrt to get m/sqrt(Hz). OR:
    // optionally, multiply by 4/L^2, where L is in m,
    // to get 1/Hz. Take the square root to get something in units of
    // 1/sqrt(Hz).

    const double kB = 1.3806488e-23; // J/K
    const double T = 300; // K
    const double fourKbToverPi = (4.*kB*T)/M_PI;
    
    const double toSI = 1.e-6; //gives noise in m/sqrt(Hz)

    double subLossAngle = 0.;
    double coatLossAngle = 0.;

    switch(mWhichSubstrateYijkl) {
        case(kAlGaAs):
            subLossAngle = lossPhi_AlGaAs;
            break;
        case(kIso_AlGaAs):
            subLossAngle = lossPhi_Iso_AlGaAs;
            break;
        case(kIso_Ta2O5):
            subLossAngle = lossPhi_Iso_Ta2O5;
            break;
        case(kIso_FusedSilica):
            subLossAngle = lossPhi_Iso_FusedSilica;
            break;
        default:
            std::cout << "WARNING! Invalid Yijkl string. Do not trust results!\n";
            break;
    }
        
    switch(mWhichCoatingYijkl) {
        case(kAlGaAs):
            coatLossAngle = lossPhi_AlGaAs;
            break;
        case(kIso_AlGaAs):
            coatLossAngle = lossPhi_Iso_AlGaAs;
            break;
        case(kIso_Ta2O5):
            coatLossAngle = lossPhi_Iso_Ta2O5;
            break;
        case(kIso_FusedSilica):
            coatLossAngle = lossPhi_Iso_FusedSilica;
            break;
        default:
            std::cout << "WARNING! Invalid Yijkl string. Do not trust results!\n";
            break;
    }

    const double subNoiseTimesf = 
      toSI * fourKbToverPi * substrateEnergy * subLossAngle;
    const double coatNoiseTimesf = 
      //toSI * fourKbToverPi * coatingEnergy * lossPhi_AlGaAs;
      //Use same loss angle for coating whether crystal or effective isotropic
      toSI * fourKbToverPi * coatingEnergy * coatLossAngle;
    const double totalNoiseTimesf = subNoiseTimesf + coatNoiseTimesf;

    const double subAmpNoiseTimesSqrtf = sqrt(subNoiseTimesf);
    const double coatAmpNoiseTimesSqrtf = sqrt(coatNoiseTimesf);
    const double totalAmpNoiseTimesSqrtf = sqrt(totalNoiseTimesf);
    

    pcout << "Total energy: " << energy << std::endl;
    pcout << "Substrate energy: " << substrateEnergy << std::endl;
    pcout << "Coating energy: " << coatingEnergy << std::endl;

    pcout << std::endl;
    pcout << "Total amp noise (m/sqrt(Hz)) * sqrt(f/Hz): " 
	  << totalAmpNoiseTimesSqrtf 
	  << std::endl;
    pcout << "Substrate amp noise (m/sqrt(Hz)) * sqrt(f/Hz): " 
	  << subAmpNoiseTimesSqrtf 
	  << std::endl;
    pcout << "Coating amp noise (m/sqrt(Hz)) * sqrt(f/Hz): " 
	  << coatAmpNoiseTimesSqrtf 
	  << std::endl;

    // print if column 0
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      std::ofstream eOut("Energy.dat", std::ios::app);
      //print headers before resolution 0
      if (cycle == 0) {
	eOut << "# Elastic energy in deformed mirror " << std::endl;       
	eOut << "# [1] refinement level = cycle" << std::endl;
	eOut << "# [2] energy" << std::endl;
	eOut << "# [3] substrate energy" << std::endl;
	eOut << "# [4] coating energy" << std::endl;
	eOut << "# [5] substrate amp noise (m/sqrt(Hz)) * sqrt(f)" 
	     << std::endl;
	eOut << "# [6] coating amp noise (m/sqrt(Hz)) * sqrt(f)" << std::endl;
	eOut << "# [7] total amp noise (m/sqrt(Hz)) * sqrt(f)" << std::endl;
      }
      eOut << std::setprecision(14);
      eOut << cycle << " " << energy << " " 
	   << substrateEnergy << " " << coatingEnergy << " "
	   << subAmpNoiseTimesSqrtf << " " << coatAmpNoiseTimesSqrtf << " " 
	   << totalAmpNoiseTimesSqrtf
	   << std::endl;
    }

  }
    
  // run() sets up the grid and then solves the problem
  // The grid must be coarse but fine enough to pick up the key behavior
  // of the solution; otherwise, you just get the trivial solution zero back.
  template <int dim>
  void ElasticProblem<dim>::run ()
  {
    for (unsigned int cycle=0; cycle<mNumberOfCycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;
	pcout << "Time: " << currentDateTime() << std::endl;

        if (cycle == 0) //set up the grid
          {
	    // Make a cylinder
	    GridGenerator::cylinder(triangulation, rad, halflength);

	    // The cylinder by default has an axis on the x axis.
	    // This rotates and shifts the cylinder so the coating is above 
	    // z=0 and so the cylinder axis is the z axis.
	    GridTools::transform(CylTransFunc(d,halflength),triangulation); 

	    //The manifold helps code know how to refine to get smoother
	    //culinder instead of starting with an octagonal prism and then
	    //refining that, instead of increasing the number of prism sides
	    static const CylindricalManifold<dim> cylindrical_manifold(2);
	    triangulation.set_all_manifold_ids(0);
	    triangulation.set_manifold(0,cylindrical_manifold);


	    pcout << "Grid created. Refinement level: " 
		  << triangulation.n_global_levels() << "."
		  << std::endl;


	    //start out with some basic refinement for a very coarse grid
            triangulation.refine_global (2);

	    pcout << "Grid globally refined (2x). Refinement level: " 
	    	  << triangulation.n_global_levels() << "."
	    	  << std::endl;

	    //refine to put more points near the boundary
	    //This is necessary for the solver to latch on to the 
	    //correct solution.
	    const int refineByDIts = 2;//in the past, I used 6 here

	    for (unsigned int j=0; j<refineByDIts; ++j) {
	      typename Triangulation<dim>::active_cell_iterator
		cell = triangulation.begin_active(),
		endc = triangulation.end();
	      for (; cell!=endc; ++cell) {
		for (unsigned int v=0;
		     v < GeometryInfo<dim>::vertices_per_cell;
		     ++v) {
		  const double vertexDist = cell->vertex(v)(dim-1);
		  double vertexRad = 0.;
		  for (unsigned int k=0; k<dim-1; ++k) {
		    vertexRad += cell->vertex(v)(k)*cell->vertex(v)(k);
		  }
		  vertexRad = sqrt(vertexRad);
		  if (fabs(vertexDist) < r0
		      && vertexRad < (refineByDIts-j)*r0) {
		    cell->set_refine_flag();
		    break;
		  }
		}
	      }
	      pcout << "Grid refinement " << j << "...";
	    triangulation.execute_coarsening_and_refinement();
	    pcout << "done. Refinement level: "
		  << triangulation.n_global_levels() << "."
		  << std::endl;
	    }	  

	    //Set the Neumann boundary...top face: find by looping over all
            typename Triangulation<dim>::cell_iterator
	      cell = triangulation.begin (),
	      endc = triangulation.end();
            for (; cell!=endc; ++cell)
              for (unsigned int face_number=0;
                   face_number<GeometryInfo<dim>::faces_per_cell;
                   ++face_number)
	    	//if the cell face has a center that is at z=d,
	    	//this is the Neumann boundary
                if (d - cell->face(face_number)->center()(dim-1) 
	    	    < 1.e-12) {		  
                  cell->face(face_number)->set_boundary_id (1);
	    	}
	  }
        else //not first cycle
          refine_grid ();

	//Solve the problem for this cycle ("cycle" = "resolution")

        pcout << "   Number of active cells:       "
                  << triangulation.n_global_active_cells()
                  << std::endl;

        setup_system ();

        pcout << "   Number of degrees of freedom: "
                  << dof_handler.n_dofs()
                  << std::endl;

        assemble_system ();
        solve ();
        output_results (cycle);

        computing_timer.print_summary ();
        computing_timer.reset ();

	pcout << std::endl;

      }
  }
}


int main (int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace QuasistaticBrownianThermalNoise;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Braces makes sure ElasticProblem goes out of scope, destructs
      // before PETScFinalize is called.
      {
	dealii::deallog.depth_console (0);
	ElasticProblem<3> elastic_problem_3d;
	elastic_problem_3d.run ();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
