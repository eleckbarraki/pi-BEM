

#include "../include/bem_problem.h"

#include <deal.II/numerics/error_estimator.h>

#include <iomanip>
#include <iostream>
#include <memory>

#include "../include/laplace_kernel.h"
#include "../include/singular_kernel_integral.h"
#include "../include/quasi_singular_kernel_integral.h"
#include "../include/sinh_quadrature.h"
#include "../include/telles_quadrature.h"
#include "../include/subdivision_quadrature.h"
#include "Teuchos_TimeMonitor.hpp"

using Teuchos::RCP;
using Teuchos::Time;
using Teuchos::TimeMonitor;
using namespace std;

#define ENTRY EntryRaiiObject obj##LINE(__FUNCTION__);

struct EntryRaiiObject
{
  EntryRaiiObject(const char *f)
    : f_(f)
  {
    printf("Entered into %s\n", f_);
  }
  ~EntryRaiiObject()
  {
    printf("Exited from %s\n", f_);
  }
  const char *f_;
};
namespace
{
  template <typename VEC>
  void
  vector_shift(VEC &in_vec, double a_scalar)
  {
    for (auto i : in_vec.locally_owned_elements())
      in_vec[i] += a_scalar;
  }
} // namespace
RCP<Time> ConstraintsTime =
  Teuchos::TimeMonitor::getNewTimer("Compute Constraints Time");
RCP<Time> AssembleTime = Teuchos::TimeMonitor::getNewTimer("Assemble Time");
RCP<Time> NormalsTime  = Teuchos::TimeMonitor::getNewTimer("Normals Time");
RCP<Time> SurfaceGradientTime =
  Teuchos::TimeMonitor::getNewTimer("SurfaceGradientTime Time");
RCP<Time> GradientTime = Teuchos::TimeMonitor::getNewTimer("Gradient Time");
RCP<Time> LacSolveTime = Teuchos::TimeMonitor::getNewTimer("LAC Solve Time");
RCP<Time> ReinitTime =
  Teuchos::TimeMonitor::getNewTimer("BEM Reinitialisation Time");

// @sect4{BEMProblem::BEMProblem and
// BEMProblem::read_parameters}
// The constructor initializes the
// variuous object in much the same
// way as done in the finite element
// programs such as step-4 or
// step-6. The only new ingredient
// here is the ParsedFunction object,
// which needs, at construction time,
// the specification of the number of
// components.
//
// For the exact solution the number
// of vector components is one, and
// no action is required since one is
// the default value for a
// ParsedFunction object. The wind,
// however, requires dim components
// to be specified. Notice that when
// declaring entries in a parameter
// file for the expression of the
// Functions::ParsedFunction, we need
// to specify the number of
// components explicitly, since the
// function
// Functions::ParsedFunction::declare_parameters
// is static, and has no knowledge of
// the number of components.
template <>
BEMProblem<3>::BEMProblem(ComputationalDomain<3> &comp_dom,
                          // const unsigned int fe_degree,
                          MPI_Comm comm)
  : pcout(std::cout)
  , comp_dom(comp_dom)
  , dh(comp_dom.tria)
  , gradient_dh(comp_dom.tria)
  , mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}
template <>
BEMProblem<2>::BEMProblem(ComputationalDomain<2> &comp_dom,
                          // const unsigned int fe_degree,
                          MPI_Comm comm)
  : pcout(std::cout)
  , comp_dom(comp_dom)
  , dh(comp_dom.tria)
  , gradient_dh(comp_dom.tria)
  , mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}


namespace
{
  template <int dim, int spacedim, bool system = false>
  std::unique_ptr<FiniteElement<dim, spacedim>>
  createFiniteElementPointer(const std::string &type, const unsigned int order)
  {
    if (type == "FEQ")
      {
        if (system)
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FESystem<dim, spacedim>(FE_Q<dim, spacedim>(order),
                                          spacedim));
          }
        else
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FE_Q<dim, spacedim>(order));
          }
      }
    else
      {
        if (system)
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FESystem<dim, spacedim>(FE_DGQ<dim, spacedim>(order),
                                          spacedim));
          }
        else
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FE_DGQ<dim, spacedim>(order));
          }
      }
  }
} // namespace
template <int dim>
void
BEMProblem<dim>::reinit()
{
  // ENTRY
  Teuchos::TimeMonitor LocalTimer(*ReinitTime);

  fe          = createFiniteElementPointer<dim - 1, dim, false>(scalar_fe_type,
                                                       scalar_fe_order);
  gradient_fe = createFiniteElementPointer<dim - 1, dim, true>(vector_fe_type,
                                                               vector_fe_order);
  // fe = new FE_DGQArbitraryNodes<dim-1, dim>(QGauss<1> (2));
  // gradient_fe = new
  // FESystem<dim-1,dim>(FE_DGQArbitraryNodes<dim-1,dim>(QGauss<1> (2)),dim);
  // // auto hhh = new FE_DGQArbitraryNodes<dim-1, dim>(QGauss<1> (2));
  std::string foo = fe->get_name();
  std::cout << foo << std::endl;
  // FiniteElement<dim-1,dim> * pippo = FETools::get_fe_by_name<dim-1,
  // dim>(foo); std::cout<<pippo->get_name()<<std::endl;

  dh.distribute_dofs(*fe);
  gradient_dh.distribute_dofs(*gradient_fe);

  // we should choose the appropriate renumbering strategy and then stick with
  // it. in step 32 they use component_wise which is very straight-forward but
  // maybe the quickest is subdomain_wise (step 17, 18)
  DoFRenumbering::component_wise(dh);
  DoFRenumbering::component_wise(gradient_dh);

  pcout << "re-ordering vector" << std::endl;

  compute_reordering_vectors();

  DoFRenumbering::subdomain_wise(dh);
  DoFRenumbering::subdomain_wise(gradient_dh);

  vector_constraints.reinit();
  DoFTools::make_hanging_node_constraints(gradient_dh, vector_constraints);
  vector_constraints.close();
  if (mapping_type == "FE")
    {
      map_vector.reinit(gradient_dh.n_dofs());
      // Fills the euler vector with information from the Triangulation
      VectorTools::get_position_vector(gradient_dh, map_vector);
      vector_constraints.distribute(map_vector);
    }
  // mapping_degree = fe->get_degree();
  if (!mapping)
    {
      if (comp_dom.spheroid_bool && comp_dom.used_spherical_manifold)
        {
          for (types::global_dof_index ii = 0; ii < gradient_dh.n_dofs() / dim;
               ++ii)
            {
              map_vector[vec_original_to_sub_wise[ii]] *=
                comp_dom.spheroid_x_axis;
              map_vector[vec_original_to_sub_wise[ii + gradient_dh.n_dofs() /
                                                         dim]] *=
                comp_dom.spheroid_y_axis;
              if (dim == 3)
                map_vector[vec_original_to_sub_wise[ii + gradient_dh.n_dofs() /
                                                           dim]] *=
                  comp_dom.spheroid_z_axis;
            }
        }
      if (mapping_type == "FE")
        mapping = std::make_shared<MappingFEField<dim - 1, dim>>(gradient_dh,
                                                                 map_vector);
      else
        mapping = std::make_shared<MappingQ<dim - 1, dim>>(mapping_degree);
    }



  const types::global_dof_index n_dofs = dh.n_dofs();

  pcout << dh.n_dofs() << " " << gradient_dh.n_dofs() << std::endl;
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);

  DoFTools::get_subdomain_association(dh, dofs_domain_association);
  std::vector<types::subdomain_id> vector_dofs_domain_association(
    gradient_dh.n_dofs());

  DoFTools::get_subdomain_association(gradient_dh,
                                      vector_dofs_domain_association);


  this_cpu_set.clear();
  vector_this_cpu_set.clear();
  this_cpu_set.set_size(n_dofs);
  vector_this_cpu_set.set_size(gradient_dh.n_dofs());


  // We compute this two vector in order to use an eventual
  // DoFRenumbering::subdomain_wise At the time being we don't. We need to
  // decide the better strategy.


  // We need to enforce consistency between the non-ghosted IndexSets.
  // To be changed accordingly with the DoFRenumbering strategy.
  pcout << "you are using " << sizeof(dh.n_dofs()) << " bytes indices"
        << std::endl;
  pcout << "setting cpu_sets" << std::endl;

  for (types::global_dof_index i = 0; i < n_dofs; ++i)
    if (dofs_domain_association[i] == this_mpi_process)
      {
        this_cpu_set.add_index(i);
        types::global_dof_index dummy = sub_wise_to_original[i];
        for (unsigned int idim = 0; idim < dim; ++idim)
          {
            vector_this_cpu_set.add_index(
              vec_original_to_sub_wise[gradient_dh.n_dofs() / dim * idim +
                                       dummy]);
          }
      }


  // for (unsigned int i=0; i<gradient_dh.n_dofs(); ++i)
  //   if (vector_dofs_domain_association[i] == this_mpi_process)
  //     {
  //       vector_this_cpu_set.add_index(i);
  //       // for(unsigned int idim=0; idim<dim; ++idim)
  //       // {
  //       //   vector_this_cpu_set.add_index(i*dim+idim);
  //       // }
  //     }

  this_cpu_set.compress();
  vector_this_cpu_set.compress();
  // std::cout<<"set the cpu sets"<<std::endl;
  // std::vector<types::global_dof_index> localized_ndfos(n_mpi_processes);
  // std::vector<types::global_dof_index>
  // localized_vector_ndfos(n_mpi_processes); start_per_process.resize
  // (n_mpi_processes); vector_start_per_process.resize (n_mpi_processes);
  //
  // localized_ndfos[this_mpi_process] = this_cpu_set.n_elements();
  // localized_vector_ndfos[this_mpi_process] =
  // vector_this_cpu_set.n_elements();
  //
  // Utilities::MPI::sum (localized_ndfos, mpi_communicator, start_per_process);
  // Utilities::MPI::sum (localized_vector_ndfos, mpi_communicator,
  // vector_start_per_process);
  //
  // for(unsigned int i=start_per_process.size()-1; i>0; --i)
  // {
  //   start_per_process[i] = start_per_process[i-1];
  //   vector_start_per_process[i] = vector_start_per_process[i-1];
  // }
  // start_per_process[0] = 0;
  // vector_start_per_process[0] = 0;
  // for(unsigned int i=2; i<start_per_process.size(); ++i)
  // {
  //   start_per_process[i] += start_per_process[i-1];
  //   vector_start_per_process[i] += vector_start_per_process[i-1];
  // }
  // start_per_process[0] = 0;
  // vector_start_per_process[0] = 0;

  // std::cout<<this_mpi_process<<" "<<start_per_process[this_mpi_process]<<"
  // "<<vector_start_per_process[this_mpi_process]<<std::endl;


  // At this point we just need to create a ghosted IndexSet for the scalar
  // DoFHandler. This can be through the builtin dealii function.
  // this_cpu_set.print(std::cout);
  MPI_Barrier(mpi_communicator);
  ghosted_set.clear();
  ghosted_set.set_size(dh.n_dofs());
  ghosted_set =
    DoFTools::dof_indices_with_subdomain_association(dh, this_mpi_process);
  ghosted_set.compress();
  // std::cout<<"set ghosted set"<<std::endl;

  // standard TrilinosWrappers::MPI::Vector reinitialization.
  system_rhs.reinit(this_cpu_set, mpi_communicator);
  sol.reinit(this_cpu_set, mpi_communicator);
  alpha.reinit(this_cpu_set, mpi_communicator);
  serv_phi.reinit(this_cpu_set, mpi_communicator);
  serv_dphi_dn.reinit(this_cpu_set, mpi_communicator);
  serv_tmp_rhs.reinit(this_cpu_set, mpi_communicator);


  // TrilinosWrappers::SparsityPattern for the BEM matricesreinitialization
  pcout << "re-initializing sparsity patterns and matrices" << std::endl;
  if (solution_method == "Direct")
    {
      full_sparsity_pattern.reinit(this_cpu_set, mpi_communicator);

      for (auto i : this_cpu_set)
        {
          for (types::global_dof_index j = 0; j < dh.n_dofs(); ++j)
            full_sparsity_pattern.add(i, j);
        }

      full_sparsity_pattern.compress();
      neumann_matrix.reinit(full_sparsity_pattern);
      dirichlet_matrix.reinit(full_sparsity_pattern);
    }
  pcout << "re-initialized sparsity patterns and matrices" << std::endl;
  preconditioner_band = 100;
  preconditioner_sparsity_pattern.reinit(this_cpu_set,
                                         mpi_communicator,
                                         (types::global_dof_index)
                                           preconditioner_band);
  is_preconditioner_initialized = false;

  dirichlet_nodes.reinit(this_cpu_set, mpi_communicator);
  neumann_nodes.reinit(this_cpu_set, mpi_communicator);
  compute_dirichlet_and_neumann_dofs_vectors();
  compute_double_nodes_set();

  fma.init_fma(dh,
               double_nodes_set,
               dirichlet_nodes,
               *mapping,
               quadrature_order,
               singular_quadrature_order);



  // We need a TrilinosWrappers::MPI::Vector to reinit the SparsityPattern for
  // the parallel mass matrices.
  TrilinosWrappers::MPI::Vector helper(vector_this_cpu_set, mpi_communicator);
  // These are just for test
  // IndexSet vector_active_dofs;
  // IndexSet vector_relevant_dofs;
  IndexSet trial_index_set;
  // vector_active_dofs.clear();
  // vector_relevant_dofs.clear();
  trial_index_set.clear();
  // DoFTools::extract_locally_active_dofs(gradient_dh, vector_active_dofs);//,
  // vector_active_dofs);
  trial_index_set =
    DoFTools::dof_indices_with_subdomain_association(gradient_dh,
                                                     this_mpi_process);
  // Assert(trial_index_set == vector_this_cpu_set, ExcNotImplemented());
  // // The following functions returns the entire dof set.
  // DoFTools::extract_locally_relevant_dofs(gradient_dh, vector_relevant_dofs);
  // pcout<<vector_active_dofs.n_elements()<<"
  // "<<vector_relevant_dofs.n_elements()<<"
  // "<<vector_this_cpu_set.n_elements()<<std::endl;


  // This is the only way we could create the SparsityPattern, through the
  // Epetramap of an existing vector.
  // vector_sparsity_pattern.reinit(helper.vector_partitioner(),
  // helper.vector_partitioner());
  vector_sparsity_pattern.reinit(vector_this_cpu_set,
                                 vector_this_cpu_set,
                                 mpi_communicator);
  DoFTools::make_sparsity_pattern(gradient_dh,
                                  vector_sparsity_pattern,
                                  vector_constraints,
                                  true,
                                  this_mpi_process);
  vector_sparsity_pattern.compress();


  hyp_alpha.reinit(this_cpu_set, mpi_communicator);
  C_ij.resize(dim * dim);
  for (unsigned int i = 0; i < dim * dim; ++i)
    C_ij[i].reinit(this_cpu_set, mpi_communicator);
  b_i.resize(dim);
  for (unsigned int i = 0; i < dim; ++i)
    b_i[i].reinit(this_cpu_set, mpi_communicator);
}

// rotate a cell and check validity of triangles
namespace
{
  // rotate the cell
  template <int dim>
  void rotate_cell(const typename DoFHandler<dim-1, dim>::active_cell_iterator &cell, 
                    const std::vector<Point<dim>> &support_points_local, 
                    const Point<dim> &singularity,std::vector<Point<dim>> &rotated_points, 
                    Point<dim> &rotated_singularity,
                    Tensor<2,dim> &rotation_matrix)
  {
    const unsigned int n = support_points_local.size();
    if (rotated_points.size() != n)
      rotated_points.resize(n);
    
    // compute middle point of the cell  
    Point<dim> P = cell->center();
//    for (unsigned int i = 0; i < n; ++i)
//      P += support_points_local[i];
//    P /= n;
    
    // compute the new y-axis direction: u = -(P - O)/|P - O| 
    const Point<dim> &S = singularity;
    const double norm_PS = (P-S).norm();
    AssertThrow(norm_PS > 1e-14, ExcMessage("Cell midpoint at origin: undefined rotation direction."));
    Tensor<1,dim> u = -(P-S) / norm_PS; // new y-axis (unit)

    // choose a vector not parallel to u
    Tensor<1,dim> a;
    a[0] = 0; a[1] = 1; a[2] = 0;   // original y
    Tensor<1,dim> cross_au = cross_product_3d(a,u);
    if (cross_au.norm() < 1e-8)
    {
      a[0] = 0; a[1] = 0; a[2] = 1;   // if u is nearly parallel to y, use z
    }

    // compute orthonormal basis (v,u,w)
    Tensor<1,dim> v = cross_product_3d(a,u);
    v /= v.norm(); // new x-axis
    Tensor<1,dim> w = cross_product_3d(v,u); // new z-axis (already unit)

    // build rotation matrix E = [v u w]
    Tensor<2,dim> E;
    for (unsigned int i=0; i < dim; ++i)
    {
      E[i][0] = v[i];
      E[i][1] = u[i];
      E[i][2] = w[i];
    }
    rotation_matrix = E;

    // rotate cell dofs
    for (unsigned int i=0; i < n; ++i)
    {
      const Point<dim> &X = support_points_local[i];
      Point<dim> Xnew;
      for (unsigned int k=0; k<dim; ++k)
      {
        Xnew[k] = 0.0;
        for (unsigned int j=0; j<dim; ++j)
          Xnew[k] += E[j][k] * (X[j]-S[j]); // E oppure Et?
      }
      rotated_points[i] = Xnew; 
    }
    
    // rotate also singularity
    
    Point<dim> Snew;
    for (unsigned int k=0; k<dim; ++k)
    {
      Snew[k] = 0.0;
//      for (unsigned int j=0; j<dim; ++j)
//        Snew[k] += E[j][k] * S[j];
    }
    rotated_singularity = Snew;
  
    return;
  }
  
  
  //check if the triangle is valid
  template <int dim>
  bool triangle_is_valid(const std::vector<Point<dim>> &v, double tol=1e-12)
  {
    // coincident points
    if ((v[1]-v[0]).norm() < tol) return false;
    if ((v[2]-v[0]).norm() < tol) return false;
    if ((v[2]-v[1]).norm() < tol) return false;

    // collinearity
    Tensor<1,dim> a = v[1] - v[0];
    Tensor<1,dim> b = v[2] - v[0];
    
    double area_tria;
    
    if(dim==2)
      area_tria = std::abs(a[0]*b[1] - a[1]*b[0]);
    else if(dim==3)
      area_tria = cross_product_3d(a, b).norm();
    else
      AssertThrow(false, ExcNotImplemented());
    
    return area_tria >= tol;
  }
  
}

// Compute α(x) by integrating dG/dn over the entire boundary with respect
// to the dof x.
//
// Compare the results obtained using:
//   - Spherical quadrature,
//   - Telles quadrature,
//   - Sinh quadrature,
//   - Subdivision quadrature.
//
//For singular cells, use the QDuffy in the Telles, sinh and
//subdivision approaches.

// Spherical quadrature layout:
//1) obtain the spherical coordinates of all the cell dofs and the cell vertices
//2) create a local triangulation with the one cell and cell vertices spherical coordinates
//3) create a dh on the new local tria
//4) use the local dofs coordinate for the local mapping on the new tria/dh
//5) create and FEValues on the new dh

template <int dim>
double BEMProblem<dim>::compute_boundary_area_with_spherical_coordinates()
{
  std::ofstream file("quadrature_points.csv", std::ios::out | std::ios::trunc);
  file << "x y z" << "\n";
  double area = 0.0;

  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);
  const unsigned int n_q_points = fe_v.n_quadrature_points;

  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);                                                  

  cell_it cell = dh.begin_active(), endc = dh.end();
  
  double tot_area_cart = 0.0;
  double tot_area_sph = 0.0;
  double tot_area_sinh = 0.0;
  double tot_area_subdivision = 0.0;
  double tot_area_telles = 0.0;

  // Choose a reproducible singularity close to a geometric edge. The closest
  // support point is selected so the test remains stable under refinement.
  // singularity4: (0.96875, 0.0605469, 0.00625);
  // singularity5: (0.984375, 0.0307617, 0.003125);
  Point<dim> target_singularity;
  if (dim == 3)
    target_singularity = Point<dim>(0.0, 0.0, -1.003);
  else if (dim == 2)
    target_singularity = Point<dim>(0.625, 0.04);

  const Vector<double> localized_hyp_alpha(hyp_alpha);
  // hyp_aplha is the free coeff computed with mantich formula

  types::global_dof_index sing_index = 0;
  double min_singularity_distance =  std::numeric_limits<double>::max();
  bool found_smooth_near_edge_dof = false;

  for (types::global_dof_index i = 0; i < support_points.size(); ++i)
  {
    const double distance = support_points[i].distance(target_singularity);
    if (distance < min_singularity_distance)
    {
      min_singularity_distance = distance;
      sing_index               = i;
      found_smooth_near_edge_dof = true;
    }
  }

  Point<dim> singularity = support_points[sing_index];
  const double correct_geom_alpha = localized_hyp_alpha[sing_index];

  // initialise quantities of interest
  Vector<double> cartesian_integral_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> spherical_integral_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> telles_integral_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> sinh_integral_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> subdivision_integral_per_cell(dh.get_triangulation().n_active_cells());
  
  Vector<double> absolute_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> relative_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> telles_absolute_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> telles_relative_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> sinh_absolute_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> sinh_relative_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> subdivision_absolute_error_per_cell(dh.get_triangulation().n_active_cells());
  Vector<double> subdivision_relative_error_per_cell(dh.get_triangulation().n_active_cells());
  
  Vector<double> sinh_abs_error_minus_telles_abs_error(dh.get_triangulation().n_active_cells());
  Vector<double> subdivision_abs_error_minus_telles_abs_error(dh.get_triangulation().n_active_cells());
  Vector<double> max_abs_error_cell_flag(dh.get_triangulation().n_active_cells());
  Vector<double> spherical_normal_alignment_error_per_cell(dh.get_triangulation().n_active_cells());
  
  Vector<double> quasi_singular_cell_flag(dh.get_triangulation().n_active_cells());
  Vector<double> singular_cell_flag(dh.get_triangulation().n_active_cells());
  
  Vector<double> distance_to_singularity_over_cell_diameter(dh.get_triangulation().n_active_cells());
  
  Vector<double> relative_error_vs_smooth_field(dh.get_triangulation().n_active_cells());
  Vector<double> relative_error_vs_geom_alpha_field(dh.get_triangulation().n_active_cells());

  // loop on cells
  for (cell = dh.begin_active(); cell != endc; ++cell)
  {
    fe_v.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    
    const std::vector<Point<dim>> &q_points    = fe_v.get_quadrature_points();
    const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();
      
    //1) we obtain the spherical coordinates of all the cell dofs and the cell vertices
    //std::cout<<"# "<<cell<<"Supp Cart:  "<<std::endl;
    std::vector<Point<dim> > spher_local_supp_points(fe->dofs_per_cell);
    
    if (dim==2)
    {
      AssertThrow(dim == 3, ExcMessage("Not yet implemented for dim = 2"));
    }
    
    bool sing_on_cell = false;
    bool quasi_sing_cell = false;
    
    //translation and matrices for rotation initializing here
    std::vector<Tensor<2,dim>> Rotations;
    Rotations.clear();
    std::vector<Tensor<2,dim>> QRotations;
    QRotations.clear();
    std::vector<Tensor<2,dim>> Rots;
    Rots.clear();
    Point<dim> translation(0.0,0.0,0.0);

    using Triangle = std::vector<Point<dim>>;
    std::vector<Triangle> subtriangles(4);
    
    using SubCell = std::vector<Point<dim>>;    // quadrilateral
    std::vector<SubCell> subcells;              // variable number of subcells 1 2 3 or 4 
    
    Point<dim> sing_to_use(0.0,0.0,0.0);
    Point<dim-1> qsing_to_use(0.0,0.0);
    double distance_to_singularity_ratio = std::numeric_limits<double>::infinity();
    
    if constexpr (dim==3)
    {          
      // define local support points to act only on one cell
      std::vector<Point<dim>> local_support_points(fe->dofs_per_cell);
      for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
        local_support_points[i] = support_points[ local_dof_indices[i] ];
      std::vector<Point<dim>> points_to_use(fe->dofs_per_cell);    
      
      // check if the singularity is on the cell
      unsigned int singular_local_index = numbers::invalid_unsigned_int;
      for (unsigned int jj=0; jj<fe->dofs_per_cell; ++jj)
      {       
        if((singularity-support_points[local_dof_indices[jj]]).norm() < 1e-14)
        {
          sing_on_cell = true;
          singular_local_index = jj;
          break;
        }    
      }
      
      // check if the cell is quasi singular: if it's close enough to singularity
      if(!sing_on_cell)
      {
        double dist_to_center = (singularity - cell->center()).norm();
        double h = cell->diameter();
        if(dist_to_center - 0.5*h < 0.5)
        {
          QuasiSingularKernelIntegral<dim> qski(cell, *fe, *mapping, singularity);
          qsing_to_use = qski.get_closest_reference_point();
          double dist_to_cell = qski.min_distance;
          
          distance_to_singularity_ratio = dist_to_cell / cell->diameter();
          if(distance_to_singularity_ratio < 15)
            quasi_sing_cell = true;
        }
      }

//      // printing the cartesian points
//      for (unsigned int jj=0; jj<fe->dofs_per_cell; ++jj)
//      {
//        Point<dim> P(local_support_points[jj]-singularity);
//        std::cout<<std::setprecision(8)<<"#"<<P<<std::endl; 
//      }
      
      // if singularity is on cell split the cell      
      if(sing_on_cell)
      {
        points_to_use = local_support_points;
        sing_to_use = singularity;
        
        // TODO: add the points to subtriangles: to each their own following the scheme
        // S A B (P SP) AB SA SB SAB
        // instert an if on FE degree o ci sono trucchi?
        
        // creating subtriangles
        subtriangles[0] = {{ sing_to_use, points_to_use[0], points_to_use[1] }};
        subtriangles[1] = {{ sing_to_use, points_to_use[1], points_to_use[3] }};
        subtriangles[2] = {{ sing_to_use, points_to_use[3], points_to_use[2] }};
        subtriangles[3] = {{ sing_to_use, points_to_use[2], points_to_use[0] }};
        
        // loop on triangles 
        for (unsigned int i = 0; i < subtriangles.size(); ++i)
        {        
          if(!triangle_is_valid(subtriangles[i]))
          {
            continue;
          }
          
          // move singularity in the origin
          for (unsigned int jj=1; jj < subtriangles[i].size(); ++jj)
            subtriangles[i][jj] -= sing_to_use; 
          
          // rotate valid triangle
          const Point<dim> &A = subtriangles[i][1];
          const Point<dim> &B = subtriangles[i][2];
          Tensor<1,dim> ex = A / A.norm(); // new x-axis
          
          // compute orthonormal basis (ex,ey,ez)
          Tensor<1,dim> ez = cross_product_3d(ex,B);
          ez /= ez.norm(); // new z-axis
          Tensor<1,dim> ey = cross_product_3d(ez, ex); // new y-axis

          // build rotation matrix E = [ex ey ez]
          Tensor<2,dim> E;
          for (unsigned int ii=0; ii < dim; ++ii)
          {
            E[0][ii] = ex[ii];
            E[1][ii] = ey[ii];
            E[2][ii] = ez[ii];
          }
          
          // save rotation and translation
          translation = sing_to_use;
          Rotations.push_back(E);
          //std::cout << "rotation matrix: \n" << E << "\n" << Rotations.back() << std::endl;
          
          for (unsigned int jj=1; jj < subtriangles[i].size(); ++jj)      // rotate subtriangle vertices 1 and 2
          {
            const Point<dim> &X = subtriangles[i][jj];
            Point<dim> Xnew;
            for (unsigned int kk=0; kk<dim; ++kk)
            {
              Xnew[kk] = 0.0;
              for (unsigned int tt=0; tt<dim; ++tt)
                Xnew[kk] += E[kk][tt] * X[tt];
            }
            subtriangles[i][jj] = Xnew; 
          }
          
          //1) convert to spherical the vertices 1 and 2
          for (unsigned int jj=1; jj < subtriangles[i].size(); ++jj)
          {
            Point<dim> spher;
            Point<dim> cart(subtriangles[i][jj]);
            
            double r = cart.norm();
            double theta = acos(cart(2)/r);
            double phi = std::atan2(cart(1), cart(0));
                
            spher(0)=r; spher(1)=theta;
            if (dim==3)
              spher(2)=phi;
            subtriangles[i][jj] = spher;
          }
      
          //1.1) fix the jump acros -pi and pi for phi   
          for (unsigned int jj = 1; jj < subtriangles[i].size(); ++jj)
          {
            double &phi = subtriangles[i][jj](2);
            double prev = subtriangles[i][jj-1](2);

            while (phi - prev > numbers::PI)
              phi -= 2.0 * numbers::PI;

            while (phi - prev < -numbers::PI)
              phi += 2.0 * numbers::PI;
          }
          
          // modify first point of trinangle to (0, thetaA, phiA)
          subtriangles[i][0](0) = 0.0;
          subtriangles[i][0](1) = subtriangles[i][1](1);
          subtriangles[i][0](2) = subtriangles[i][1](2);
          
          //add a point to valid triangles (S, A, B) --> (S, A, B, P)
          // std::cout<<"#creating quadrilateral subcell nr. "<< i <<std::endl;
          Point<dim> P(subtriangles[i][0](0), subtriangles[i][2](1), subtriangles[i][2](2));
          SubCell this_cell;
          this_cell = {{subtriangles[i][0], subtriangles[i][1], subtriangles[i][2], P}};
          subcells.push_back(this_cell);
          
//          // print subcell points
//          std::cout<<"Subcell nr. "<< i << " dofs: "<<std::endl;        
//          for (unsigned int jj=0; jj < subcells.back().size(); ++jj)
//          {
//            std::cout<<subcells.back()[jj]<<std::endl;
//          }
          
        }// end loop triangles    
      } //end if(sing_on_cell)
      else if(quasi_sing_cell)   // if the cell is quasi singular use spherical with the center in the projection point (TODO: implement in proportional measure)      
      {            
        Point<dim> closest_point = mapping->transform_unit_to_real_cell(cell, qsing_to_use);
        //std::cout << "Projection on the quasi singular cell: " << closest_point << std::endl;

        points_to_use = local_support_points;
        sing_to_use = closest_point;
        
        // save translation
        translation = sing_to_use;
        
        // creating subtriangles
        subtriangles[0] = {{ sing_to_use, points_to_use[0], points_to_use[1] }};
        subtriangles[1] = {{ sing_to_use, points_to_use[1], points_to_use[3] }};
        subtriangles[2] = {{ sing_to_use, points_to_use[3], points_to_use[2] }};
        subtriangles[3] = {{ sing_to_use, points_to_use[2], points_to_use[0] }};
        
        // loop on triangles 
        for (unsigned int i = 0; i < subtriangles.size(); ++i)
        {        
          if(!triangle_is_valid(subtriangles[i]))
          {
            // std::cout<<"#subcell nr. " << i << " is invalid, skip it "<<std::endl;
            continue;
          }
          
          // move singularity in the origin
          for (unsigned int jj=1; jj < subtriangles[i].size(); ++jj)
            subtriangles[i][jj] -= sing_to_use; 
          
          // rotate valid triangle
          const Point<dim> &A = subtriangles[i][1];
          const Point<dim> &B = subtriangles[i][2];
          Tensor<1,dim> ex = A / A.norm(); // new x-axis
          
          // compute orthonormal basis (ex,ey,ez)
          Tensor<1,dim> ez = cross_product_3d(ex,B);
          ez /= ez.norm(); // new z-axis
          Tensor<1,dim> ey = cross_product_3d(ez, ex); // new y-axis

          // build rotation matrix E = [ex ey ez]
          Tensor<2,dim> E;
          for (unsigned int ii=0; ii < dim; ++ii)
          {
            E[0][ii] = ex[ii];
            E[1][ii] = ey[ii];
            E[2][ii] = ez[ii];
          }
          
          //saving rotations here
          QRotations.push_back(E);
          //std::cout << "rotation matrix: \n" << E << "\n" << Rotations.back() << std::endl;
          
          for (unsigned int jj=1; jj < subtriangles[i].size(); ++jj)      // rotate subtriangle vertices 1 and 2
          {
            const Point<dim> &X = subtriangles[i][jj];
            Point<dim> Xnew;
            for (unsigned int kk=0; kk<dim; ++kk)
            {
              Xnew[kk] = 0.0;
              for (unsigned int tt=0; tt<dim; ++tt)
                Xnew[kk] += E[kk][tt] * X[tt];
            }
            subtriangles[i][jj] = Xnew; 
          }
          
          //1) convert to spherical the vertices 1 and 2
          for (unsigned int jj=1; jj < subtriangles[i].size(); ++jj)
          {
            Point<dim> spher;
            Point<dim> cart(subtriangles[i][jj]);
            
            double r = cart.norm();
            double theta = acos(cart(2)/r);
            double phi = std::atan2(cart(1), cart(0));
                
            spher(0)=r; spher(1)=theta;
            if (dim==3)
              spher(2)=phi;
            subtriangles[i][jj] = spher;
          }
      
          //1.1) fix the jump acros -pi and pi for phi   
          for (unsigned int jj = 1; jj < subtriangles[i].size(); ++jj)
          {
            double &phi = subtriangles[i][jj](2);
            double prev = subtriangles[i][jj-1](2);

            while (phi - prev > numbers::PI)
              phi -= 2.0 * numbers::PI;

            while (phi - prev < -numbers::PI)
              phi += 2.0 * numbers::PI;
          }
          
          // modify first point of trinangle to (0, thetaA, phiA)
          subtriangles[i][0](0) = 0.0;
          subtriangles[i][0](1) = subtriangles[i][1](1);
          subtriangles[i][0](2) = subtriangles[i][1](2);
          
          //add a point to valid triangles (S, A, B) --> (S, A, B, P)
          //std::cout<<"#creating quadrilateral subcell nr. "<< i <<std::endl;
          Point<dim> P(subtriangles[i][0](0), subtriangles[i][2](1), subtriangles[i][2](2));
          SubCell this_cell;
          this_cell = {{subtriangles[i][0], subtriangles[i][1], subtriangles[i][2], P}};
          subcells.push_back(this_cell);
          
//          // print subcell points
//          std::cout<<"Subcell nr. "<< i << " dofs: "<<std::endl;        
//          for (unsigned int jj=0; jj < subcells.back().size(); ++jj)
//          {
//            std::cout<<subcells.back()[jj]<<std::endl;
//          }
          
        }// end loop triangles

      } // end if(quasi_sing_cell)
      else   // normal cell: rotate and convert to spherical
      {
        Tensor <2,dim> E;        
        points_to_use.resize(fe->dofs_per_cell);
        rotate_cell<dim>(cell, local_support_points, singularity, points_to_use, sing_to_use,E);
        
        // save rotation and translation
        Rots.push_back(E);
        translation = singularity;
        
        //1) convert to spherical
        for (unsigned int j=0; j<fe->dofs_per_cell; ++j)
        {
          Point<dim> spher;
          Point<dim> cart(points_to_use[j]-sing_to_use);  //coordinates traslated to singularity in r = 0.0

          // here we print the (eventually rotated) coordinates of the dofs support points
          // std::cout<<std::setprecision(8)<<cart<<std::endl;
          
          double r = cart.norm();
          double theta = acos(cart(2)/r);
          double phi = std::atan2(cart(1), cart(0));
              
          spher(0)=r; spher(1)=theta;
          if (dim==3)
            spher(2)=phi;
          spher_local_supp_points[j] = spher;
        }
    
        //1.1) fix the jump acros -pi and pi for phi   
        for (unsigned int j = 1; j < fe->dofs_per_cell; ++j)
        {
          double &phi = spher_local_supp_points[j](2);
          double prev = spher_local_supp_points[j-1](2);

          while (phi - prev > numbers::PI)
            phi -= 2.0 * numbers::PI;

          while (phi - prev < -numbers::PI)
            phi += 2.0 * numbers::PI;
        }
        
//        // print the spherical coordinates computed
//        std::cout<<cell<<"#  Supp Spher:  "<<std::endl;
//        for (unsigned int j=0; j<fe->dofs_per_cell; ++j)
//        {
//          std::cout<<spher_local_supp_points[j]<<std::endl;
//        }
        
        // create one single subcell
        subcells.resize(1);
        subcells[0] = spher_local_supp_points;
        
      } // end else
    } // end dim==3   
    
    double spher_cell_area = 0.0;
    double telles_cell_area = 0.0;
    double sinh_cell_area  = 0.0;
    double subdivision_cell_area = 0.0;
    double max_spherical_normal_alignment_error = 0.0;
    for(unsigned int c = 0; c<subcells.size(); ++c)
    {  
      //2) create a one cell triangulation with the one cell and cell vertices spherical coordinates
      std::vector<Point<dim>>        spher_vertices;
      std::vector<CellData<dim - 1>> spher_cells;
      SubCellData                    spher_subcelldata;
      
      spher_vertices.resize(4);
      spher_cells.resize(1);
      
      if(sing_on_cell || quasi_sing_cell)
      {
        spher_vertices[0] = subcells[c][0];
        spher_vertices[1] = subcells[c][1];
        spher_vertices[2] = subcells[c][3];
        spher_vertices[3] = subcells[c][2];
        
        spher_cells[0].vertices[0]  = 0;
        spher_cells[0].vertices[1]  = 1;
        spher_cells[0].vertices[2]  = 2;
        spher_cells[0].vertices[3]  = 3; 
      }
      else
      {
        for (unsigned int j=0; j<GeometryInfo<dim - 1>::vertices_per_cell; ++j)
          spher_vertices[j] = subcells[c][j];       //qui
        
        spher_cells[0].vertices[0]  = 0;
        spher_cells[0].vertices[1]  = 1;
        spher_cells[0].vertices[2]  = 2;
        spher_cells[0].vertices[3]  = 3;  
      }
      
      Triangulation<dim - 1, dim> spher_tria;
      GridTools::delete_unused_vertices(spher_vertices, spher_cells, spher_subcelldata);
      GridTools::consistently_order_cells(spher_cells);
      spher_tria.create_triangulation(spher_vertices, spher_cells, spher_subcelldata);
        
      //3) create a dh and a gradient_dh on the new one cell tria    
      DoFHandler<dim - 1, dim>  spher_dh(spher_tria);
      DoFHandler<dim - 1, dim>  spher_gradient_dh(spher_tria);
      spher_dh.distribute_dofs(*fe);
      spher_gradient_dh.distribute_dofs(*gradient_fe);
      DoFRenumbering::component_wise(spher_dh);
      DoFRenumbering::component_wise(spher_gradient_dh);
         
      //4) prepare the vector with the local --- polar --- coordinates
      //   for the one cell mapping on the new tria/dh
      Vector<double>  spher_map_vector(spher_gradient_dh.n_dofs());
      std::shared_ptr<Mapping<dim - 1, dim>> spher_mapping;          
      if (mapping_type == "FE")
        spher_mapping = std::make_shared<MappingFEField<dim - 1, dim>>(spher_gradient_dh,
                                                                   spher_map_vector);
      else
        spher_mapping = std::make_shared<MappingQ<dim - 1, dim>>(mapping_degree);
          
      if(sing_on_cell || quasi_sing_cell)
      {
        const std::vector<unsigned int> ref_to_subcell = {0, 1, 3, 2};
        
        for (unsigned int j=0; j<fe->dofs_per_cell; ++j)    //TODO: now works only for linear fe
        {
          const Point<dim> &p =
            (subcells[c].size() == ref_to_subcell.size() ?
               subcells[c][ref_to_subcell[j]] :
               subcells[c][j]);
          
          spher_map_vector(j+0*fe->dofs_per_cell) = p(0);
          spher_map_vector(j+1*fe->dofs_per_cell) = p(1);
          spher_map_vector(j+2*fe->dofs_per_cell) = p(2);//spher_local_supp_points[j](2);
        }
      }
      else
      {
        for (unsigned int j=0; j<fe->dofs_per_cell; ++j)    //TODO: now works only for linear fe
        {
          spher_map_vector(j+0*fe->dofs_per_cell) = subcells[c][j](0);
          spher_map_vector(j+1*fe->dofs_per_cell) = subcells[c][j](1);
          spher_map_vector(j+2*fe->dofs_per_cell) = subcells[c][j](2);//spher_local_supp_points[j](2);
        }
      }
      
      //5) create and FEValues on the new dh
      FEValues<dim - 1, dim> spher_fe_v(*spher_mapping,
                                          *fe,
                                          *quadrature,
                                          update_values | update_gradients | update_normal_vectors |
                                          update_jacobians | update_quadrature_points | update_JxW_values);
        
        
      //6) loop on quadrature nodes to compute cell area 
      // both in standard way and with polar coordinates                                  
      cell_it spher_cell = spher_dh.begin_active(); 
      spher_fe_v.reinit(spher_cell);
      const std::vector<Point<dim>> &spher_q_points = spher_fe_v.get_quadrature_points();
      
      // compute subcell area
      double spher_subcell_area = 0.0;
      
      Tensor<2,dim> RT;
      if (sing_on_cell)
        RT = transpose(Rotations[c]);
      else if (quasi_sing_cell)
        RT = transpose(QRotations[c]);
      else
        RT = Rots[c];
      
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double r = spher_q_points[q](0);
        double theta = spher_q_points[q](1);
        double phi = spher_q_points[q](2);
        
        Tensor<2,dim> Js;
        Js[0][0] = sin(theta) * cos(phi); Js[0][1] = r * cos(theta) * cos(phi); Js[0][2] = -r * sin(theta) * sin(phi);  
        Js[1][0] = sin(theta) * sin(phi); Js[1][1] = r * cos(theta) * sin(phi); Js[1][2] = r * sin(theta) * cos(phi);
        Js[2][0] = cos(theta); Js[2][1] = -r * sin(theta); Js[2][2] = 0;
        
        Tensor<2,dim> Js_inv_T;   // Js inverso trasposto
        Js_inv_T[0][0] = sin(theta)*cos(phi); Js_inv_T[0][1] = cos(theta)*cos(phi)/r; Js_inv_T[0][2] = -sin(phi)/(r*sin(theta));
        Js_inv_T[1][0] = sin(theta)*sin(phi); Js_inv_T[1][1] = cos(theta)*sin(phi)/r; Js_inv_T[1][2] = cos(phi)/(r*sin(theta));
        Js_inv_T[2][0] = cos(theta); Js_inv_T[2][1] = -sin(theta)/r; Js_inv_T[2][2] = 0;
        
        DerivativeForm<1, dim-1, dim> Jrtf_uv = spher_fe_v.jacobian(q);
        Tensor<1,dim> rtf_u;
        Tensor<1,dim> rtf_v;
        for(unsigned int ii = 0; ii < dim; ++ii)
        {
          rtf_u[ii] = Jrtf_uv[ii][0];
          rtf_v[ii] = Jrtf_uv[ii][1];
        }
        
        // Nanson formula for area cell
        Tensor<1,dim> NN = cross_product_3d(rtf_u,rtf_v);
        Tensor<1,dim> Js_inv_T_NN = Js_inv_T * NN; 
        double area_contrib = std::sin(theta) *Js_inv_T_NN.norm();
        // double area_cell = r*r*std::sin(theta) *Js_inv_T_NN.norm();  

        // compute double layer potential in spherical coordinates
        Js_inv_T_NN = Js_inv_T_NN / Js_inv_T_NN.norm();
        double dGdn = ( Js_inv_T_NN[0] * std::sin(theta) * std::cos(phi) + Js_inv_T_NN[1] * std::sin(theta) * std::sin(phi) + Js_inv_T_NN[2] * std::cos(theta) ) 
                        / (4*numbers::PI); // r*r
        
        double spher_weight = quadrature->weight(q);
        spher_subcell_area += dGdn * area_contrib * spher_weight;       
        
//        // print spherical quadrature points in spherical coordinates
//        std::cout << "# Spherical quad points: " << std::endl;
//        std::cout<<spher_q_points[q]<<std::endl;

        // error on the normal to compare with cartesian
        Tensor<1, dim> spherical_normal_original = RT * Js_inv_T_NN;
        const double normal_alignment = spherical_normal_original * normals[q];
        max_spherical_normal_alignment_error =
          std::max(max_spherical_normal_alignment_error,
                   1.0 - std::abs(normal_alignment));
 
      } // end loop on quadrature nodes
      spher_cell_area += spher_subcell_area;

      // save spherical quadrature points in cartesian coordinates on file
      std::vector<Point<dim>> quadrature_points_rotated;
      quadrature_points_rotated.resize(n_q_points);      
      for(unsigned int q = 0; q < n_q_points; ++q)
      {
        // convert to cartesian each quadrature point
        double r = spher_q_points[q](0);
        double theta = spher_q_points[q](1);
        double phi = spher_q_points[q](2);
        quadrature_points_rotated[q][0] = r * std::sin(theta) * std::cos(phi);
        quadrature_points_rotated[q][1] = r * std::sin(theta) * std::sin(phi);
        quadrature_points_rotated[q][2] = r * std::cos(theta);
      
        //rotate each node with RT and shift with singularity
        quadrature_points_rotated[q] = RT * quadrature_points_rotated[q];
        quadrature_points_rotated[q] += translation;
        // print points
        //std::cout<<quadrature_points_rotated[q]<<std::endl;
        
        // save quadrature points inside a file
        //file << quadrature_points_rotated[q] << "\n";
      }    
          
    } //end loop subcells
    
    // compute cell area with 3 other versions for quasi sing cells (telles, sinh, subdivision)
    // use duffy for singular cells
    double sing_subcell_area = 0.0;
    double telles_subcell_area = 0.0;
    double sinh_subcell_area  = 0.0;
    double subdivision_subcell_area = 0.0;
    if constexpr (dim == 3)   // this is needed to make telles quadrature work, TODO telles quadrature for dim 1
    {
      if(quasi_sing_cell)
      {
      
        // compute area with diy telles quadrature   
        unsigned int n_telles = quadrature_order;
        Quadrature<2> telles_quad = telles_quadrature(
                                      cell,
                                      *mapping,
                                      singularity,
                                      qsing_to_use,
                                      n_telles,
                                      2  // alpha=2 for 1/r^2 kernel
                                      );
        FEValues<2,3> telles_fe_v(*mapping, *fe, telles_quad,
                                  update_values | update_gradients | update_normal_vectors |
                                  update_jacobians | update_quadrature_points | update_JxW_values);
        telles_fe_v.reinit(cell);               
        const unsigned int n_q_telles = telles_quad.size();
        const std::vector<Point<dim>>    &telles_q_points  = telles_fe_v.get_quadrature_points();
        const std::vector<Tensor<1,dim>> &telles_normals = telles_fe_v.get_normal_vectors(); 
        
        for (unsigned int q = 0; q < n_q_telles; ++q)
        {
          Tensor<1, dim> RR = telles_q_points[q]-singularity; //distanza euclidea tra xq e x0;
          Point<dim> DD;
          double     ss;
          LaplaceKernel::kernels(RR, DD, ss);
          double dGdn = DD * telles_normals[q];
      
          telles_subcell_area += -dGdn * telles_fe_v.JxW(q);
          
          file << telles_q_points[q] << "\n";
        }

        // compute area with sinh quadrature
        QSinh<dim-1> sinh_quad(n_telles, qsing_to_use);
        FEValues<dim-1, dim> sinh_fe_v(*mapping,
                                        *fe,
                                        sinh_quad,
                                        update_values | update_gradients |
                                        update_normal_vectors |
                                        update_jacobians |
                                        update_quadrature_points |
                                        update_JxW_values);
        sinh_fe_v.reinit(cell);
        const unsigned int n_q_sinh = sinh_quad.size();
        const std::vector<Point<dim>> &sinh_q_points =
          sinh_fe_v.get_quadrature_points();
        const std::vector<Tensor<1, dim>> &sinh_normals =
          sinh_fe_v.get_normal_vectors();

        for (unsigned int q = 0; q < n_q_sinh; ++q)
        {
          Tensor<1, dim> RR = sinh_q_points[q] - singularity;
          Point<dim> DD;
          double     ss;
          LaplaceKernel::kernels(RR, DD, ss);
          double dGdn = DD * sinh_normals[q];

          sinh_subcell_area += -dGdn * sinh_fe_v.JxW(q);
          
          //file << sinh_q_points[q] << "\n";
        }

        // compute area with subdivision quadrature
        QSubdivision<dim-1> subdivision_quad(n_telles, qsing_to_use);
        FEValues<dim-1, dim> subdivision_fe_v(*mapping,
                                               *fe,
                                               subdivision_quad,
                                               update_values |
                                               update_gradients |
                                               update_normal_vectors |
                                               update_jacobians |
                                               update_quadrature_points |
                                               update_JxW_values);
        subdivision_fe_v.reinit(cell);
        const unsigned int n_q_subdivision = subdivision_quad.size();
        const std::vector<Point<dim>> &subdivision_q_points =
          subdivision_fe_v.get_quadrature_points();
        const std::vector<Tensor<1, dim>> &subdivision_normals =
          subdivision_fe_v.get_normal_vectors();

        for (unsigned int q = 0; q < n_q_subdivision; ++q)
        {
          Tensor<1, dim> RR = subdivision_q_points[q] - singularity;
          Point<dim> DD;
          double     ss;
          LaplaceKernel::kernels(RR, DD, ss);
          double dGdn = DD * subdivision_normals[q];

          subdivision_subcell_area += -dGdn * subdivision_fe_v.JxW(q);
          
          //file << subdivision_q_points[q] << "\n";
        }
          
      } // end if(quasi_sing_cell)
      else if(sing_on_cell)
      {
        // use qsplit and qduffy
        //    QDuffy(n, beta): n = quadrature order, beta = 1.0 standard for 1/R singularities
        //    QSplit automatically splits the reference cell into triangles
        //    all with vertex zero at ref_sing, then applies QDuffy to each
        Point<dim-1> ref_sing = mapping->transform_real_to_unit_cell(cell, singularity);
        unsigned int n_duffy = singular_quadrature_order;
        QDuffy duffy_quad(n_duffy, 1.0);
        QSplit<dim-1> split_quad(duffy_quad, ref_sing);
        FEValues<dim-1, dim> sing_fe_v(*mapping, *fe, split_quad,
                                        update_values | update_gradients | update_normal_vectors |
                                        update_jacobians | update_quadrature_points | update_JxW_values);
        sing_fe_v.reinit(cell);
        const unsigned int n_q_sing = split_quad.size();
        const std::vector<Point<dim>> &sing_q_points = sing_fe_v.get_quadrature_points();
        const std::vector<Tensor<1, dim>> &sing_normals = sing_fe_v.get_normal_vectors();
        
        for (unsigned int q = 0; q < n_q_sing; ++q)
        {
          Tensor<1, dim> RR = sing_q_points[q] - singularity;
          Point<dim> DD;
          double     ss;
          LaplaceKernel::kernels(RR, DD, ss);
          double dGdn = DD * sing_normals[q];

          sing_subcell_area += -dGdn * sing_fe_v.JxW(q);
          
          file << sing_q_points[q] << "\n";
        }
        
      }// end if (sing_on_cell)
    } // end if(dim == 3)
    telles_cell_area += (quasi_sing_cell) ? telles_subcell_area : (sing_on_cell) ? sing_subcell_area : spher_cell_area;
    sinh_cell_area += (quasi_sing_cell) ? sinh_subcell_area : (sing_on_cell) ? sing_subcell_area : spher_cell_area;
    subdivision_cell_area += (quasi_sing_cell) ? subdivision_subcell_area : (sing_on_cell) ? sing_subcell_area : spher_cell_area; 
    
    // compute cartesian cell area
    double cell_area = 0.0; 
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      Tensor<1, dim> RR = q_points[q]-singularity; //distanza euclidea tra xq e x0;
      Point<dim> DD;
      double     ss;
      LaplaceKernel::kernels(RR, DD, ss);
      double dGGdn = DD * normals[q];
      
      cell_area += dGGdn * fe_v.JxW(q);
      
      if((!sing_on_cell) || (!quasi_sing_cell))
        file << q_points[q] << "\n";
    }
    
    //save data cell by cell
    const unsigned int cell_data_index = cell->active_cell_index();
    const double       cartesian_contribution = -cell_area;
    const double       denominator = std::max(std::abs(cartesian_contribution), 1e-14);                                 
                                          
    cartesian_integral_per_cell[cell_data_index]    = cartesian_contribution;
    spherical_integral_per_cell[cell_data_index]    = spher_cell_area;
    telles_integral_per_cell[cell_data_index]       = telles_cell_area;
    sinh_integral_per_cell[cell_data_index]         = sinh_cell_area;
    subdivision_integral_per_cell[cell_data_index]  = subdivision_cell_area;
    
    absolute_error_per_cell[cell_data_index] = std::abs(spher_cell_area - cartesian_contribution);
    relative_error_per_cell[cell_data_index] = absolute_error_per_cell[cell_data_index] / denominator;
    telles_absolute_error_per_cell[cell_data_index] = std::abs(telles_cell_area - cartesian_contribution);
    telles_relative_error_per_cell[cell_data_index] = telles_absolute_error_per_cell[cell_data_index] / denominator;
    sinh_absolute_error_per_cell[cell_data_index] = std::abs(sinh_cell_area - cartesian_contribution);
    sinh_relative_error_per_cell[cell_data_index] = sinh_absolute_error_per_cell[cell_data_index] / denominator;
    subdivision_absolute_error_per_cell[cell_data_index] = std::abs(subdivision_cell_area - cartesian_contribution);
    subdivision_relative_error_per_cell[cell_data_index] = subdivision_absolute_error_per_cell[cell_data_index] / denominator;
    
    sinh_abs_error_minus_telles_abs_error[cell_data_index] =
      sinh_absolute_error_per_cell[cell_data_index] - telles_absolute_error_per_cell[cell_data_index];
    subdivision_abs_error_minus_telles_abs_error[cell_data_index] =
      subdivision_absolute_error_per_cell[cell_data_index] - telles_absolute_error_per_cell[cell_data_index];

    spherical_normal_alignment_error_per_cell[cell_data_index] =
      max_spherical_normal_alignment_error;
      
    quasi_singular_cell_flag[cell_data_index] = quasi_sing_cell ? 1.0 : 0.0;
    singular_cell_flag[cell_data_index]       = sing_on_cell ? 1.0 : 0.0;
    
    distance_to_singularity_over_cell_diameter[cell_data_index] =
      std::isfinite(distance_to_singularity_ratio) ? distance_to_singularity_ratio : -1.0;
       
    // total error estimators
    tot_area_cart += cell_area;
    tot_area_sph += spher_cell_area;
    tot_area_telles += telles_cell_area;
    tot_area_sinh += sinh_cell_area;
    tot_area_subdivision += subdivision_cell_area; 
  } // end loop on cells

  // relative errors wrt value of alpha
  relative_error_vs_smooth_field =  std::abs(tot_area_sph - 0.5) / 0.5;
  relative_error_vs_geom_alpha_field =  std::abs(tot_area_sph - correct_geom_alpha) / correct_geom_alpha;

  double max_absolute_cell_error  = 0.0;
  double max_relative_cell_error  = 0.0;
  double mean_absolute_cell_error = 0.0;
  double mean_relative_cell_error = 0.0;
  double max_telles_absolute_cell_error  = 0.0;
  double max_telles_relative_cell_error  = 0.0;
  double mean_telles_absolute_cell_error = 0.0;
  double mean_telles_relative_cell_error = 0.0;
  double max_sinh_absolute_cell_error  = 0.0;
  double max_sinh_relative_cell_error  = 0.0;
  double mean_sinh_absolute_cell_error = 0.0;
  double mean_sinh_relative_cell_error = 0.0;
  double max_subdivision_absolute_cell_error  = 0.0;
  double max_subdivision_relative_cell_error  = 0.0;
  double mean_subdivision_absolute_cell_error = 0.0;
  double mean_subdivision_relative_cell_error = 0.0;
  double max_spherical_normal_alignment_error = 0.0;
  unsigned int n_quasi_singular_cells = 0;
  unsigned int n_singular_cells       = 0;
  unsigned int max_abs_error_cell_index             = 0;
  unsigned int max_telles_abs_error_cell_index      = 0;
  unsigned int max_sinh_abs_error_cell_index        = 0;
  unsigned int max_subdivision_abs_error_cell_index = 0;

  //  This code loops over all cells and extracts:
  //  maximum errors,
  //  mean errors,
  //  indices of the worst cells,
  //  counts of special cells,
  //  counts of which Telles rule was used.
  for (unsigned int i = 0; i < absolute_error_per_cell.size(); ++i)
    {
      if (absolute_error_per_cell[i] > max_absolute_cell_error)
      {
        max_absolute_cell_error = absolute_error_per_cell[i];
        max_abs_error_cell_index = i;
      }
      max_relative_cell_error =  std::max(max_relative_cell_error, relative_error_per_cell[i]);
      mean_absolute_cell_error += absolute_error_per_cell[i];
      mean_relative_cell_error += relative_error_per_cell[i];
      
      if (telles_absolute_error_per_cell[i] > max_telles_absolute_cell_error)
      {
        max_telles_absolute_cell_error = telles_absolute_error_per_cell[i];
        max_telles_abs_error_cell_index = i;
      }
      max_telles_relative_cell_error = std::max(max_telles_relative_cell_error, telles_relative_error_per_cell[i]);
      mean_telles_absolute_cell_error += telles_absolute_error_per_cell[i];
      mean_telles_relative_cell_error += telles_relative_error_per_cell[i];
      
      if (sinh_absolute_error_per_cell[i] > max_sinh_absolute_cell_error)
      {
        max_sinh_absolute_cell_error = sinh_absolute_error_per_cell[i];
        max_sinh_abs_error_cell_index = i;
      }
      max_sinh_relative_cell_error = std::max(max_sinh_relative_cell_error, sinh_relative_error_per_cell[i]);
      mean_sinh_absolute_cell_error += sinh_absolute_error_per_cell[i];
      mean_sinh_relative_cell_error += sinh_relative_error_per_cell[i];
      
      if (subdivision_absolute_error_per_cell[i] > max_subdivision_absolute_cell_error)
      {
        max_subdivision_absolute_cell_error = subdivision_absolute_error_per_cell[i];
        max_subdivision_abs_error_cell_index = i;
      }
      max_subdivision_relative_cell_error = std::max(max_subdivision_relative_cell_error, subdivision_relative_error_per_cell[i]);
      mean_subdivision_absolute_cell_error += subdivision_absolute_error_per_cell[i];
      mean_subdivision_relative_cell_error += subdivision_relative_error_per_cell[i];
      
      max_spherical_normal_alignment_error = std::max(max_spherical_normal_alignment_error,
                 spherical_normal_alignment_error_per_cell[i]);
                 
      n_quasi_singular_cells += (quasi_singular_cell_flag[i] > 0.5 ? 1u : 0u);
      n_singular_cells += (singular_cell_flag[i] > 0.5 ? 1u : 0u);
    }

  max_abs_error_cell_flag[max_abs_error_cell_index]             = 1.0;
  max_abs_error_cell_flag[max_telles_abs_error_cell_index]      = 1.0;
  max_abs_error_cell_flag[max_sinh_abs_error_cell_index]        = 1.0;
  max_abs_error_cell_flag[max_subdivision_abs_error_cell_index] = 1.0;

  if (absolute_error_per_cell.size() > 0)
    {
      mean_absolute_cell_error /= absolute_error_per_cell.size();
      mean_relative_cell_error /= relative_error_per_cell.size();
      mean_telles_absolute_cell_error /= telles_absolute_error_per_cell.size();
      mean_telles_relative_cell_error /= telles_relative_error_per_cell.size();
      mean_sinh_absolute_cell_error /= sinh_absolute_error_per_cell.size();
      mean_sinh_relative_cell_error /= sinh_relative_error_per_cell.size();
      mean_subdivision_absolute_cell_error /= subdivision_absolute_error_per_cell.size();
      mean_subdivision_relative_cell_error /= subdivision_relative_error_per_cell.size();
    }

  const double cartesian_total = -tot_area_cart;
  const double geom_denominator = std::max(std::abs(correct_geom_alpha), 1e-14);
  const double smooth_error = std::abs(tot_area_sph - 0.5) / 0.5;
  const double geom_error = std::abs(tot_area_sph - correct_geom_alpha) / geom_denominator;
  const double telles_smooth_error = std::abs(tot_area_telles - 0.5) / 0.5;
  const double telles_geom_error = std::abs(tot_area_telles - correct_geom_alpha) / geom_denominator;
  const double sinh_smooth_error = std::abs(tot_area_sinh - 0.5) / 0.5;
  const double sinh_geom_error = std::abs(tot_area_sinh - correct_geom_alpha) / geom_denominator;
  const double subdivision_smooth_error = std::abs(tot_area_subdivision - 0.5) / 0.5;
  const double subdivision_geom_error = std::abs(tot_area_subdivision - correct_geom_alpha) / geom_denominator;
  const double cartesian_geom_error = std::abs(cartesian_total - correct_geom_alpha) / geom_denominator;

  // lambda function to print worst cells diagnostics
  auto print_cell_error_details = [&](const std::string &label,
                                      const unsigned int index) {
    pcout << "  " << label << " max abs error cell:" << std::endl
          << "    cell index: " << index << std::endl
          << "    cartesian integral: " << cartesian_integral_per_cell[index]
          << std::endl
          << "    spherical integral: " << spherical_integral_per_cell[index]
          << std::endl
          << "    Telles integral: " << telles_integral_per_cell[index]
          << std::endl
          << "    sinh integral: " << sinh_integral_per_cell[index]
          << std::endl
          << "    subdivision integral: " << subdivision_integral_per_cell[index] 
          << std::endl
          << "    spherical abs error: " << absolute_error_per_cell[index]         
          << std::endl
          << "    telles abs error: " << telles_absolute_error_per_cell[index]
          << std::endl
          << "    sinh abs error: " << sinh_absolute_error_per_cell[index]
          << std::endl
          << "    subdivision abs error: " << subdivision_absolute_error_per_cell[index] 
          << std::endl
          << "    is singular: " << (singular_cell_flag[index] > 0.5 ? "true" : "false")
          << std::endl
          << "    is quasi singular: " << (quasi_singular_cell_flag[index] > 0.5 ? "true" : "false")
          << std::endl
          << "    distance/cell diameter: " << distance_to_singularity_over_cell_diameter[index] << std::endl
          << "    spherical normal alignment error: " << spherical_normal_alignment_error_per_cell[index]
          << std::endl;
  };

  // print overall quadrature stats
  pcout << "Spherical quadrature accuracy summary:" << std::endl
        << "  target singularity: " << target_singularity << std::endl
        << "  selected singularity: " << singularity << std::endl
        << "  singular dof index: " << sing_index << std::endl
        << "  smooth near-edge dof selected: "
        << (found_smooth_near_edge_dof ? "true" : "false") << std::endl
        << "  target selection distance: " << min_singularity_distance
        << std::endl
        << "  total cartesian integral: " << cartesian_total << std::endl
        << "  total spherical integral: " << tot_area_sph << std::endl
        << "  total telles integral: " << tot_area_telles << std::endl
        << "  total sinh integral: " << tot_area_sinh << std::endl
        << "  total subdivision integral: " << tot_area_subdivision
        << std::endl
        << "  reference smooth alpha: " << 0.5 << std::endl
        << "  reference geom_alpha: " << correct_geom_alpha << std::endl
        << "  Spherical total rel error vs smooth: " << smooth_error << std::endl
        << "  Spherical total rel error vs geom_alpha: " << geom_error
        << std::endl
        << "  Telles total rel error vs smooth: " << telles_smooth_error << std::endl
        << "  Telles total rel error vs geom_alpha: " << telles_geom_error
        << std::endl
        << "  sinh total rel error vs smooth: " << sinh_smooth_error << std::endl
        << "  sinh total rel error vs geom_alpha: " << sinh_geom_error
        << std::endl
        << "  subdivision total rel error vs smooth: " << subdivision_smooth_error << std::endl
        << "  subdivision total rel error vs geom_alpha: " << subdivision_geom_error
        << std::endl
        << "  cartesian rel error vs geom_alpha: " << cartesian_geom_error
        << std::endl
        << "  Spherical mean cell rel error vs cartesian: " << mean_relative_cell_error << std::endl
        << "  Spherical max cell rel error vs cartesian: " << max_relative_cell_error << std::endl
        << "  Spherical mean cell abs error vs cartesian: " << mean_absolute_cell_error << std::endl
        << "  Spherical max cell abs error vs cartesian: " << max_absolute_cell_error
        << std::endl
        << "  Telles mean cell rel error vs cartesian: " << mean_telles_relative_cell_error << std::endl
        << "  Telles max cell rel error vs cartesian: " << max_telles_relative_cell_error << std::endl
        << "  Telles mean cell abs error vs cartesian: " << mean_telles_absolute_cell_error << std::endl
        << "  Telles max cell abs error vs cartesian: " << max_telles_absolute_cell_error
        << std::endl
        << "  sinh mean cell rel error vs cartesian: " << mean_sinh_relative_cell_error << std::endl
        << "  sinh max cell rel error vs cartesian: " << max_sinh_relative_cell_error << std::endl
        << "  sinh mean cell abs error vs cartesian: " << mean_sinh_absolute_cell_error << std::endl
        << "  sinh max cell abs error vs cartesian: " << max_sinh_absolute_cell_error
        << std::endl
        << "  subdivision mean cell rel error vs cartesian: " << mean_subdivision_relative_cell_error << std::endl
        << "  subdivision max cell rel error vs cartesian: " << max_subdivision_relative_cell_error << std::endl
        << "  subdivision mean cell abs error vs cartesian: " << mean_subdivision_absolute_cell_error << std::endl
        << "  subdivision max cell abs error vs cartesian: " << max_subdivision_absolute_cell_error
        << std::endl
        << "  quasi-singular cells: " << n_quasi_singular_cells << std::endl
        << "  singular cells: " << n_singular_cells << std::endl
        << "  max spherical normal alignment error: "
        << max_spherical_normal_alignment_error << std::endl;

  // print worst cells diagnostics
  print_cell_error_details("Telles", max_telles_abs_error_cell_index);
  if (max_sinh_abs_error_cell_index != max_telles_abs_error_cell_index)
    print_cell_error_details("sinh", max_sinh_abs_error_cell_index);
  if (max_subdivision_abs_error_cell_index != max_telles_abs_error_cell_index &&
      max_subdivision_abs_error_cell_index != max_sinh_abs_error_cell_index)
    print_cell_error_details("subdivision", max_subdivision_abs_error_cell_index);

  // post-processing and visualization, writes quantities in spherical_quadrature_error.vtu
  if (this_mpi_process == 0)
    {
      DataOut<dim - 1, dim> dataout;
      dataout.attach_dof_handler(dh);
      dataout.add_data_vector(cartesian_integral_per_cell,
                              "cartesian_integral",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(spherical_integral_per_cell,
                              "spherical_integral",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(telles_integral_per_cell,
                              "telles_integral",
                              DataOut<dim - 1, dim>::type_cell_data);                        
      dataout.add_data_vector(sinh_integral_per_cell,
                              "sinh_integral",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(subdivision_integral_per_cell,
                              "subdivision_integral",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(absolute_error_per_cell,
                              "spherical_abs_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(relative_error_per_cell,
                              "spherical_rel_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(telles_absolute_error_per_cell,
                              "telles_abs_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(telles_relative_error_per_cell,
                              "telles_rel_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);                        
      dataout.add_data_vector(sinh_absolute_error_per_cell,
                              "sinh_abs_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(sinh_relative_error_per_cell,
                              "sinh_rel_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(subdivision_absolute_error_per_cell,
                              "subdivision_abs_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(subdivision_relative_error_per_cell,
                              "subdivision_rel_error_vs_cartesian",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(sinh_abs_error_minus_telles_abs_error,
                              "sinh_abs_error_minus_telles_abs_error",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(subdivision_abs_error_minus_telles_abs_error,
                              "subdivision_abs_error_minus_telles_abs_error",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(max_abs_error_cell_flag,
                              "is_max_abs_error_cell",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(spherical_normal_alignment_error_per_cell,
                              "spherical_normal_alignment_error",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(quasi_singular_cell_flag,
                              "is_quasi_singular_cell",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(singular_cell_flag,
                              "is_singular_cell",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(distance_to_singularity_over_cell_diameter,
                              "distance_to_singularity_over_cell_diameter",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(relative_error_vs_smooth_field,
                              "total_rel_error_vs_smooth",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.add_data_vector(relative_error_vs_geom_alpha_field,
                              "total_rel_error_vs_geom_alpha",
                              DataOut<dim - 1, dim>::type_cell_data);
      dataout.build_patches(*mapping,
                            mapping_degree,
                            DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_error("spherical_quadrature_error.vtu");
      dataout.write_vtu(file_error);
    }
return area;
}

template <>
const Quadrature<2> &
BEMProblem<3>::get_singular_quadrature(const unsigned int index) const
{
  Assert(index < fe->dofs_per_cell, ExcIndexRange(0, fe->dofs_per_cell, index));



  static std::vector<Quadrature<2>> quadratures;
  {
    if (quadratures.size() == 0)
      for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
        {
          quadratures.push_back(QSplit<2>(QDuffy(singular_quadrature_order, 1.),
                                          fe->get_unit_support_points()[i]));
        }
  }

  return quadratures[index];
}

template <>
const Quadrature<1> &
BEMProblem<2>::get_singular_quadrature(const unsigned int index) const
{
  Assert(index < fe->dofs_per_cell, ExcIndexRange(0, fe->dofs_per_cell, index));

  static std::vector<Quadrature<1>> quadratures;
  if (quadratures.size() == 0)
    for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
      {
        quadratures.push_back(QTelles<1>(singular_quadrature_order,
                                         fe->get_unit_support_points()[i]));
      }
  return quadratures[index];
}

template <int dim>
void
BEMProblem<dim>::declare_parameters(ParameterHandler &prm)
{
  // In the solver section, we set
  // all SolverControl
  // parameters. The object will then
  // be fed to the GMRES solver in
  // the solve_system() function.

  prm.enter_subsection("Solver");
  SolverControl::declare_parameters(prm);
  prm.leave_subsection();

  prm.declare_entry("Preconditioner", "ILU", Patterns::Selection("ILU|AMG"));

  prm.declare_entry("Solution method",
                    "Direct",
                    Patterns::Selection("Direct|FMA"));

  prm.enter_subsection("Quadrature rules");
  {
    prm.declare_entry("Quadrature type",
                      "gauss",
                      Patterns::Selection(
                        QuadratureSelector<(dim - 1)>::get_quadrature_names()));
    prm.declare_entry("Quadrature order", "4", Patterns::Integer());
    prm.declare_entry("Singular quadrature order", "5", Patterns::Integer());
  }
  prm.leave_subsection();

  prm.declare_entry("Mapping Type", "FE", Patterns::Selection("FE|Q"));

  prm.declare_entry("Mapping Q Degree", "1", Patterns::Integer());

  prm.declare_entry("Continuos gradient across edges",
                    "true",
                    Patterns::Bool());
  prm.enter_subsection("Scalar Finite Element");
  {
    prm.declare_entry("Finite Element type",
                      "FEQ",
                      Patterns::Selection("FEQ|FEDGQ"));
    prm.declare_entry("Finite element order", "1", Patterns::Integer());
  }
  prm.leave_subsection();

  prm.enter_subsection("Vector Finite Element");
  {
    prm.declare_entry("Finite Element type",
                      "FEQ",
                      Patterns::Selection("FEQ|FEDGQ"));
    prm.declare_entry("Finite element order", "1", Patterns::Integer());
  }
  prm.leave_subsection();


  prm.enter_subsection("Grid Refinement");
  {
    prm.declare_entry("Coarsening threshold", "0.03", Patterns::Double());
    prm.declare_entry("Refinement threshold", "0.3", Patterns::Double());
  }
  prm.leave_subsection();
}

template <int dim>
void
BEMProblem<dim>::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Solver");
  solver_control.parse_parameters(prm);
  prm.leave_subsection();

  preconditioner_type = prm.get("Preconditioner");

  solution_method = prm.get("Solution method");


  prm.enter_subsection("Quadrature rules");
  {
    quadrature = std::shared_ptr<Quadrature<dim - 1>>(
      new QuadratureSelector<dim - 1>(prm.get("Quadrature type"),
                                      prm.get_integer("Quadrature order")));
    quadrature_order          = prm.get_integer("Quadrature order");
    singular_quadrature_order = prm.get_integer("Singular quadrature order");
  }
  prm.leave_subsection();

  mapping_type       = prm.get("Mapping Type");
  mapping_degree     = prm.get_integer("Mapping Q Degree");
  continuos_gradient = prm.get_bool("Continuos gradient across edges");


  prm.enter_subsection("Scalar Finite Element");
  {
    scalar_fe_type  = prm.get("Finite Element type");
    scalar_fe_order = prm.get_integer("Finite element order");
  }
  prm.leave_subsection();

  prm.enter_subsection("Vector Finite Element");
  {
    vector_fe_type  = prm.get("Finite Element type");
    vector_fe_order = prm.get_integer("Finite element order");
  }
  prm.leave_subsection();

  prm.enter_subsection("Grid Refinement");
  {
    coarsening_threshold = prm.get_double("Coarsening threshold");
    refinement_threshold = prm.get_double("Refinement threshold");
  }
  prm.leave_subsection();
}


template <int dim>
void
BEMProblem<dim>::compute_dirichlet_and_neumann_dofs_vectors()
{
  have_dirichlet_bc = false;


  Vector<double> non_partitioned_dirichlet_nodes(dh.n_dofs());
  Vector<double> non_partitioned_neumann_nodes(dh.n_dofs());



  cell_it cell = dh.begin_active(), endc = dh.end();


  vector_shift(non_partitioned_neumann_nodes, 1.);
  std::vector<types::global_dof_index> dofs(fe->dofs_per_cell);
  std::vector<types::global_dof_index> gradient_dofs(
    gradient_fe->dofs_per_cell);
  unsigned int helper_dirichlet = 0;
  for (; cell != endc; ++cell)
    {
      if (cell->subdomain_id() == this_mpi_process)
        {
          bool dirichlet = false;
          for (auto dummy : comp_dom.dirichlet_boundary_ids)
            {
              if (dummy == cell->material_id())
                {
                  cell->get_dof_indices(dofs);
                  for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                    {
                      non_partitioned_dirichlet_nodes(dofs[i]) = 1;
                      non_partitioned_neumann_nodes(dofs[i])   = 0;
                      // pcout<<dofs[i]<<"  cellMatId "<<cell->material_id()<<"
                      // surfNodes: "<<dirichlet_nodes(dofs[i])<<"  otherNodes:
                      // "<<neumann_nodes(dofs[i])<<std::endl;
                    }
                  dirichlet        = true;
                  helper_dirichlet = 1.;
                  break;
                }
            }
          if (!dirichlet)
            {
              cell->get_dof_indices(dofs);
              // for(unsigned int i=0; i<fe->dofs_per_cell; ++i)
              // {
              //   non_partitioned_neumann_nodes(dofs[i]) = 1;
              //   non_partitioned_dirichlet_nodes(dofs[i]) = 0;
              // }
            }

          // if (cell->material_id() == comp_dom.dirichlet_sur_ID1 ||
          //     cell->material_id() == comp_dom.dirichlet_sur_ID2 ||
          //     cell->material_id() == comp_dom.dirichlet_sur_ID3)
          //   {
          //     // This is a free surface node.
          //     cell->get_dof_indices(dofs);
          //     for (unsigned int i=0; i<fe->dofs_per_cell; ++i)
          //       {
          //         non_partitioned_dirichlet_nodes(dofs[i]) = 1;
          //         non_partitioned_neumann_nodes(dofs[i]) = 0;
          //         //pcout<<dofs[i]<<"  cellMatId "<<cell->material_id()<<"
          //         surfNodes: "<<dirichlet_nodes(dofs[i])<<"  otherNodes:
          //         "<<neumann_nodes(dofs[i])<<std::endl;
          //       }
          //   }
          // else
          //   {
          //     for (unsigned int i=0; i<fe->dofs_per_cell; ++i)
          //       {
          //         cell->get_dof_indices(dofs);
          //         //pcout<<dofs[i]<<"  cellMatId "<<cell->material_id()<<"
          //         surfNodes: "<<dirichlet_nodes(dofs[i])<<"  otherNodes:
          //         "<<neumann_nodes(dofs[i])<<std::endl;
          //       }
          //
          //   }
        }
    }

  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    if (this_cpu_set.is_element(i))
      {
        dirichlet_nodes(i) = non_partitioned_dirichlet_nodes(i);
        neumann_nodes(i)   = non_partitioned_neumann_nodes(i);
      }
  // dirichlet_nodes.add(non_partitioned_dirichlet_nodes, true);// =
  // non_partitioned_dirichlet_nodes;
  // neumann_nodes.add(non_partitioned_neumann_nodes, true);// =
  // non_partitioned_neumann_nodes;
  unsigned int helper_dirichlet_2;
  // std::cout<<this_mpi_process<<" , "<<helper_dirichlet<<std::endl;
  MPI_Allreduce(&helper_dirichlet,
                &helper_dirichlet_2,
                1,
                MPI_UNSIGNED,
                MPI_MAX,
                mpi_communicator);
  // std::cout<<this_mpi_process<<" , "<<helper_dirichlet<<" ,
  // "<<helper_dirichlet_2<<std::endl;
  if (helper_dirichlet_2 > 0)
    have_dirichlet_bc = true;
  // std::cout<<this_mpi_process<<" , "<<have_dirichlet_bc<<std::endl;
  // for (unsigned int i=0; i<dh.n_dofs(); ++i)
  //    if (this_mpi_process == 1)
  //       pcout<<i<<" "<<dirichlet_nodes(i)<<" "<<neumann_nodes(i)<<std::endl;
}

template <int dim>
void
BEMProblem<dim>::compute_double_nodes_set()
{
  double tol = 1e-10;
  double_nodes_set.clear();
  double_nodes_set.resize(dh.n_dofs());
  std::vector<Point<dim>> support_points(dh.n_dofs());

  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);

  typename DoFHandler<dim - 1, dim>::active_cell_iterator cell =
                                                            dh.begin_active(),
                                                          endc = dh.end();
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  edge_set.clear();
  edge_set.set_size(dh.n_dofs());

  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          {
            cell->face(f)->get_dof_indices(face_dofs);
            for (unsigned int k = 0; k < face_dofs.size(); ++k)
              edge_set.add_index(face_dofs[k]);
          }
    }
  edge_set.compress();

  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    double_nodes_set[i].insert(i);
  for (auto i : edge_set) //(types::global_dof_index i=0; i<dh.n_dofs(); ++i)
    {
      for (auto j : edge_set)
        {
          if (support_points[i].distance(support_points[j]) < tol)
            {
              double_nodes_set[i].insert(j);
            }
        }
    }
}

template <int dim>
void
BEMProblem<dim>::compute_reordering_vectors()
{
  original_to_sub_wise.resize(dh.n_dofs());
  sub_wise_to_original.resize(dh.n_dofs());
  vec_original_to_sub_wise.resize(gradient_dh.n_dofs());
  vec_sub_wise_to_original.resize(gradient_dh.n_dofs());

  DoFRenumbering::compute_subdomain_wise(original_to_sub_wise, dh);
  DoFRenumbering::compute_subdomain_wise(vec_original_to_sub_wise, gradient_dh);

  for (types::global_dof_index i = 0; i < gradient_dh.n_dofs(); ++i)
    {
      if (i < dh.n_dofs())
        {
          sub_wise_to_original[original_to_sub_wise[i]] = i;
        }
      vec_sub_wise_to_original[vec_original_to_sub_wise[i]] = i;
    }
}
template <int dim>
void
BEMProblem<dim>::assemble_system()
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  pcout << "(Directly) Assembling system matrices" << std::endl;

  neumann_matrix   = 0;
  dirichlet_matrix = 0;



  // Next, we initialize an FEValues
  // object with the quadrature
  // formula for the integration of
  // the kernel in non singular
  // cells. This quadrature is
  // selected with the parameter
  // file, and needs to be quite
  // precise, since the functions we
  // are integrating are not
  // polynomial functions.
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  const unsigned int n_q_points = fe_v.n_quadrature_points;

  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  pcout << fe->dofs_per_cell << " " << std::endl;
  // Unlike in finite element
  // methods, if we use a collocation
  // boundary element method, then in
  // each assembly loop we only
  // assemble the information that
  // refers to the coupling between
  // one degree of freedom (the
  // degree associated with support
  // point $i$) and the current
  // cell. This is done using a
  // vector of fe->dofs_per_cell
  // elements, which will then be
  // distributed to the matrix in the
  // global row $i$. The following
  // object will hold this
  // information:
  Vector<double> local_neumann_matrix_row_i(fe->dofs_per_cell);
  Vector<double> local_dirichlet_matrix_row_i(fe->dofs_per_cell);

  // Now that we have checked that
  // the number of vertices is equal
  // to the number of degrees of
  // freedom, we construct a vector
  // of support points which will be
  // used in the local integrations:
  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);


  // After doing so, we can start the
  // integration loop over all cells,
  // where we first initialize the
  // FEValues object and get the
  // values of $\mathbf{\tilde v}$ at
  // the quadrature points (this
  // vector field should be constant,
  // but it doesn't hurt to be more
  // general):


  cell_it cell = dh.begin_active(), endc = dh.end();

  Point<dim> D;
  double     s;

  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      const std::vector<Point<dim>> &q_points    = fe_v.get_quadrature_points();
      const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();

      // We then form the integral over
      // the current cell for all
      // degrees of freedom (note that
      // this includes degrees of
      // freedom not located on the
      // current cell, a deviation from
      // the usual finite element
      // integrals). The integral that
      // we need to perform is singular
      // if one of the local degrees of
      // freedom is the same as the
      // support point $i$. A the
      // beginning of the loop we
      // therefore check wether this is
      // the case, and we store which
      // one is the singular index:
      for (types::global_dof_index i = 0; i < dh.n_dofs();
           ++i) // these must now be the locally owned dofs. the rest should
                // stay the same
        {
          if (this_cpu_set.is_element(i))
            {
              local_neumann_matrix_row_i   = 0;
              local_dirichlet_matrix_row_i = 0;

              bool         is_singular    = false;
              unsigned int singular_index = numbers::invalid_unsigned_int;

              for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                // if(local_dof_indices[j] == i)
                if (double_nodes_set[i].count(local_dof_indices[j]) > 0)
                  {
                    singular_index = j;
                    is_singular    = true;
                    break;
                  }

              // We then perform the
              // integral. If the index $i$
              // is not one of the local
              // degrees of freedom, we
              // simply have to add the
              // single layer terms to the
              // right hand side, and the
              // double layer terms to the
              // matrix:
              if (is_singular == false)
                {
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                      const Tensor<1, dim> R = q_points[q] - support_points[i];
                      LaplaceKernel::kernels(R, D, s);
                      // if(support_points[i][0]==0.25&&support_points[i][1]==0.25)
                      //   pcout<<"D "<<D<<" s "<<s<<" , ";
                      for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                        {
                          local_neumann_matrix_row_i(j) +=
                            ((D * normals[q]) * fe_v.shape_value(j, q) *
                             fe_v.JxW(q));
                          local_dirichlet_matrix_row_i(j) +=
                            (s * fe_v.shape_value(j, q) * fe_v.JxW(q));
                        }
                    }
                }
              else
                {
                  // Now we treat the more
                  // delicate case. If we
                  // are here, this means
                  // that the cell that
                  // runs on the $j$ index
                  // contains
                  // support_point[i]. In
                  // this case both the
                  // single and the double
                  // layer potential are
                  // singular, and they
                  // require special
                  // treatment.
                  //
                  // Whenever the
                  // integration is
                  // performed with the
                  // singularity inside the
                  // given cell, then a
                  // special quadrature
                  // formula is used that
                  // allows one to
                  // integrate arbitrary
                  // functions against a
                  // singular weight on the
                  // reference cell.
                  // Notice that singular
                  // integration requires a
                  // careful selection of
                  // the quadrature
                  // rules. In particular
                  // the deal.II library
                  // provides quadrature
                  // rules which are
                  // taylored for
                  // logarithmic
                  // singularities
                  // (QGaussLog,
                  // QGaussLogR), as well
                  // as for 1/R
                  // singularities
                  // (QGaussOneOverR).
                  //
                  // Singular integration
                  // is typically obtained
                  // by constructing
                  // weighted quadrature
                  // formulas with singular
                  // weights, so that it is
                  // possible to write
                  //
                  // \f[
                  //   \int_K f(x) s(x) dx = \sum_{i=1}^N w_i f(q_i)
                  // \f]
                  //
                  // where $s(x)$ is a given
                  // singularity, and the weights
                  // and quadrature points
                  // $w_i,q_i$ are carefully
                  // selected to make the formula
                  // above an equality for a
                  // certain class of functions
                  // $f(x)$.
                  //
                  // In all the finite
                  // element examples we
                  // have seen so far, the
                  // weight of the
                  // quadrature itself
                  // (namely, the function
                  // $s(x)$), was always
                  // constantly equal to 1.
                  // For singular
                  // integration, we have
                  // two choices: we can
                  // use the definition
                  // above, factoring out
                  // the singularity from
                  // the integrand (i.e.,
                  // integrating $f(x)$
                  // with the special
                  // quadrature rule), or
                  // we can ask the
                  // quadrature rule to
                  // "normalize" the
                  // weights $w_i$ with
                  // $s(q_i)$:
                  //
                  // \f[
                  //   \int_K f(x) s(x) dx =
                  //   \int_K g(x) dx = \sum_{i=1}^N \frac{w_i}{s(q_i)} g(q_i)
                  // \f]
                  //
                  // We use this second
                  // option, through the @p
                  // factor_out_singularity
                  // parameter of both
                  // QGaussLogR and
                  // QGaussOneOverR.
                  //
                  // These integrals are
                  // somewhat delicate,
                  // especially in two
                  // dimensions, due to the
                  // transformation from
                  // the real to the
                  // reference cell, where
                  // the variable of
                  // integration is scaled
                  // with the determinant
                  // of the transformation.
                  //
                  // In two dimensions this
                  // process does not
                  // result only in a
                  // factor appearing as a
                  // constant factor on the
                  // entire integral, but
                  // also on an additional
                  // integral alltogether
                  // that needs to be
                  // evaluated:
                  //
                  // \f[
                  //  \int_0^1 f(x)\ln(x/\alpha) dx =
                  //  \int_0^1 f(x)\ln(x) dx - \int_0^1 f(x) \ln(\alpha) dx.
                  // \f]
                  //
                  // This process is taken care of by
                  // the constructor of the QGaussLogR
                  // class, which adds additional
                  // quadrature points and weights to
                  // take into consideration also the
                  // second part of the integral.
                  //
                  // A similar reasoning
                  // should be done in the
                  // three dimensional
                  // case, since the
                  // singular quadrature is
                  // taylored on the
                  // inverse of the radius
                  // $r$ in the reference
                  // cell, while our
                  // singular function
                  // lives in real space,
                  // however in the three
                  // dimensional case
                  // everything is simpler
                  // because the
                  // singularity scales
                  // linearly with the
                  // determinant of the
                  // transformation. This
                  // allows us to build the
                  // singular two
                  // dimensional quadrature
                  // rules once and for all
                  // outside the loop over
                  // all cells, using only
                  // a pointer where needed.
                  //
                  // Notice that in one
                  // dimensional
                  // integration this is
                  // not possible, since we
                  // need to know the
                  // scaling parameter for
                  // the quadrature, which
                  // is not known a
                  // priori. Here, the
                  // quadrature rule itself
                  // depends also on the
                  // size of the current
                  // cell. For this reason,
                  // it is necessary to
                  // create a new
                  // quadrature for each
                  // singular
                  // integration. Since we
                  // create it using the
                  // new operator of C++,
                  // we also need to
                  // destroy it using the
                  // dual of new:
                  // delete. This is done
                  // at the end, and only
                  // if dim == 2.
                  //
                  // Putting all this into a
                  // dimension independent
                  // framework requires a little
                  // trick. The problem is that,
                  // depending on dimension, we'd
                  // like to either assign a
                  // QGaussLogR<1> or a
                  // QGaussOneOverR<2> to a
                  // Quadrature<dim-1>. C++
                  // doesn't allow this right
                  // away, and neither is a
                  // static_cast
                  // possible. However, we can
                  // attempt a dynamic_cast: the
                  // implementation will then
                  // look up at run time whether
                  // the conversion is possible
                  // (which we <em>know</em> it
                  // is) and if that isn't the
                  // case simply return a null
                  // pointer. To be sure we can
                  // then add a safety check at
                  // the end:
                  Assert(singular_index != numbers::invalid_unsigned_int,
                         ExcInternalError());

                  const Quadrature<dim - 1> *singular_quadrature =
                    &(get_singular_quadrature(singular_index));
                  Assert(singular_quadrature, ExcInternalError());

                  FEValues<dim - 1, dim> fe_v_singular(
                    *mapping,
                    *fe,
                    *singular_quadrature,
                    update_jacobians | update_values | update_normal_vectors |
                      update_quadrature_points);

                  fe_v_singular.reinit(cell);

                  const std::vector<Tensor<1, dim>> &singular_normals =
                    fe_v_singular.get_normal_vectors();
                  const std::vector<Point<dim>> &singular_q_points =
                    fe_v_singular.get_quadrature_points();

                  for (unsigned int q = 0; q < singular_quadrature->size(); ++q)
                    {
                      const Tensor<1, dim> R =
                        singular_q_points[q] - support_points[i];
                      LaplaceKernel::kernels(R, D, s);

                      for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                        {
                          local_neumann_matrix_row_i(j) +=
                            ((D * singular_normals[q]) *
                             fe_v_singular.shape_value(j, q) *
                             fe_v_singular.JxW(q));

                          local_dirichlet_matrix_row_i(j) +=
                            (s * fe_v_singular.shape_value(j, q) *
                             fe_v_singular.JxW(q));
                        }
                    }
                }

              // Finally, we need to add
              // the contributions of the
              // current cell to the
              // global matrix.
              for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                {
                  neumann_matrix.add(i,
                                     local_dof_indices[j],
                                     local_neumann_matrix_row_i(j));
                  dirichlet_matrix.add(i,
                                       local_dof_indices[j],
                                       local_dirichlet_matrix_row_i(j));
                }
            }
        }
    }

  // The second part of the integral
  // operator is the term
  // $\alpha(\mathbf{x}_i)
  // \phi_j(\mathbf{x}_i)$. Since we
  // use a collocation scheme,
  // $\phi_j(\mathbf{x}_i)=\delta_{ij}$
  // and the corresponding matrix is
  // a diagonal one with entries
  // equal to $\alpha(\mathbf{x}_i)$.

  // One quick way to compute this
  // diagonal matrix of the solid
  // angles, is to use the Neumann
  // matrix itself. It is enough to
  // multiply the matrix with a
  // vector of elements all equal to
  // -1, to get the diagonal matrix
  // of the alpha angles, or solid
  // angles (see the formula in the
  // introduction for this). The
  // result is then added back onto
  // the system matrix object to
  // yield the final form of the
  // matrix:

  /*
    pcout<<"Neumann"<<std::endl;
    for (unsigned int i = 0; i < dh.n_dofs(); i++)
        {
        if (this_cpu_set.is_element(i))
           {
           pcout<<this_mpi_process<<" *** ";
           for (unsigned int j = 0; j < dh.n_dofs(); j++)
               {
               pcout<<neumann_matrix(i,j)<<" ";
               }
           pcout<<std::endl;
           }
        }



    pcout<<"Dirichlet"<<std::endl;
    for (unsigned int i = 0; i < dh.n_dofs(); i++)
        {
        if (this_cpu_set.is_element(i))
           {
           pcout<<this_mpi_process<<" *** ";
           for (unsigned int j = 0; j < dh.n_dofs(); j++)
               {
               pcout<<dirichlet_matrix(i,j)<<" ";
               }
           pcout<<std::endl;
           }
        }
        //*/
  pcout << "done assembling system matrices" << std::endl;
  // std::cout<<"printing Neumann Matrix"<<std::endl;
  // for(unsigned int i=0; i<dh.n_dofs(); ++i)
  // {
  //   for(unsigned int j=0; j<dh.n_dofs(); ++j)
  //     std::cout<<neumann_matrix(i,j)<<" ";
  //   std::cout<<std::endl;
  // }
  // std::cout<<"printing Dirichlet Matrix"<<std::endl;
  // for(unsigned int i=0; i<dh.n_dofs(); ++i)
  // {
  //   for(unsigned int j=0; j<dh.n_dofs(); ++j)
  //     std::cout<<dirichlet_matrix(i,j)<<" ";
  //   std::cout<<std::endl;
  // }
}

template <int dim>
void
BEMProblem<dim>::compute_hypersingular_free_coeffs()
{
  pcout << "Computing free coefficients for hypersingular BIE" << std::endl;

  pcout << "Computing C_ij tensor" << endl;

  Assert(fe->has_support_points(),
         ExcMessage(
           "The FE selected has no support points. This is not supported."));
  const std::vector<Point<dim - 1>> &ref_dofs_location =
    fe->get_unit_support_points();
  // we use these points as quadrature points for a quadrature rule
  std::vector<double> weights(ref_dofs_location.size(), 1.0);


  // here's the quadrature rule obtained with the points and weights generated
  Quadrature<dim - 1> dofs_quadrature(ref_dofs_location, weights);

  // and here's the FEValues class resulting by it
  FEValues<dim - 1, dim> dofs_fe_values(*mapping,
                                        *fe,
                                        dofs_quadrature,
                                        update_values | update_gradients |
                                          update_quadrature_points |
                                          update_JxW_values |
                                          update_normal_vectors |
                                          update_jacobians |
                                          update_jacobian_grads);

  std::vector<Point<dim> > support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( *mapping, dh,
    support_points);

  cell_it cell = dh.begin_active(), endc = dh.end();
  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);


  for (types::global_dof_index i = 0; i < dh.n_dofs();
       ++i) // these must now be the locally owned dofs. the rest should stay
            // the same
    {
      std::vector<Tensor<1, dim>> normals;
      if (this_cpu_set.is_element(i))
        {
          for (cell = dh.begin_active(); cell != endc; ++cell)
            {
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i_loc = 0; i_loc < fe->dofs_per_cell; ++i_loc)
                {
                  std::set<types::global_dof_index> doubles =
                    double_nodes_set[local_dof_indices[i_loc]];
                  for (std::set<types::global_dof_index>::iterator it =
                         doubles.begin();
                       it != doubles.end();
                       it++)
                    if (*it == i)
                      {
                        dofs_fe_values.reinit(cell);
                        Tensor<1, dim> normal =
                          dofs_fe_values.normal_vector(i_loc);
                        normals.push_back(normal);
                      }
                }
            }
          // cout<<i<<"  --> "<<normals.size()<<endl;
          // for (unsigned int k=0; k<normals.size(); ++k)
          //     cout<<normals[k]<<endl;
          std::vector<Tensor<1, dim>> unique_normals;
          std::vector<Tensor<1, dim>> unique_projected_normals;

          std::vector<Tensor<1, dim>> unique_tangents;
          for (unsigned int k = 0; k < normals.size(); ++k)
            {
              bool found = false;
              for (unsigned int j = 0; j < unique_normals.size(); ++j)
                if ((normals[k] - unique_normals[j]).norm() < 1e-5)
                  found = true;
              if (!found)
                unique_normals.push_back(normals[k]);
            }
          // cout<<i<<"  -->Unique normals: "<<unique_normals.size()<<endl;
          Tensor<1, dim> average_normal;
          for (unsigned int k = 0; k < unique_normals.size(); ++k)
            average_normal += unique_normals[k] * (1.0 / unique_normals.size());
          average_normal /= average_normal.norm();
          // for (unsigned int k=0; k<unique_normals.size(); ++k)
          //     cout<<unique_normals[k]<<endl;
          Tensor<2, dim> projection_matrix =
            -1.0 * outer_product(average_normal, average_normal);
          for (unsigned int d = 0; d < dim; ++d)
            projection_matrix[d][d] += 1.0;
          for (unsigned int k = 0; k < unique_normals.size(); ++k)
            unique_projected_normals.push_back(
              projection_matrix * unique_normals[k] /
              (projection_matrix * unique_normals[k]).norm());

          // for (unsigned int k=0; k<unique_normals.size(); ++k)
          //     cout<<"n: "<<unique_normals[k]<<"  proj:
          //     "<<unique_projected_normals[k]<<endl;
          std::vector<Tensor<1, dim>> unique_normals_copy(unique_normals);
          // std::vector<Tensor<1, dim> >
          // unique_ordered_normals(unique_normals);
          std::vector<Tensor<1, dim>> unique_ordered_normals;
          // cout<<i<<"  -->Average normal: "<<average_normal<<endl;
          // cout<<i<<"  -->Proejction matrix: "<<projection_matrix<<endl;
          unsigned int N = unique_normals.size();
          unique_ordered_normals.push_back(unique_normals[0]);
          unique_normals.erase(unique_normals.begin() + 0);
          Tensor<1, dim> previous_proj_normal = unique_projected_normals[0];
          unique_projected_normals.erase(unique_projected_normals.begin() + 0);
          // unsigned int count = 0;
          while (unique_ordered_normals.size() < N)
            {
              // count++;
              // cout<<count<<" ---> "<<unique_projected_normals.size()<<endl;
              double       min_sorter = 100.0;
              double       sorter;
              unsigned int index = 0;
              // cout<<"&&& "<<unique_projected_normals.size()<<endl;
              for (unsigned int p = 0; p < unique_projected_normals.size(); ++p)
                {
                  if (dim == 3)
                    {
                      double quadrant_indicator =
                        average_normal *
                        cross_product_3d(previous_proj_normal,
                                         unique_projected_normals[p]);
                      sorter = acos(previous_proj_normal *
                                    unique_projected_normals[p]);
                      if (quadrant_indicator > 0 && sorter < min_sorter)
                        {
                          min_sorter = sorter;
                          index      = p;
                        }
                      // cout<<p<<"  "<<sorter<<endl;
                    }
                  else if (dim == 2) // this is just a dummy line to get it to
                                     // compile for dim==2 (an exception is
                                     // thrown in execution in such a case)
                    sorter = average_normal *
                             cross_product_2d(unique_ordered_normals[0]);
                }
              // cout<<"Selected: "<<index<<"  "<<endl;
              unique_ordered_normals.push_back(unique_normals[index]);
              unique_normals.erase(unique_normals.begin() + index);
              previous_proj_normal = unique_projected_normals[index];
              unique_projected_normals.erase(unique_projected_normals.begin() +
                                             index);
            }
          // for (unsigned int k=0; k<unique_ordered_normals.size(); ++k)
          //     cout<<unique_ordered_normals[k]<<"   *** vs *** "<<
          //     unique_normals_copy[k]<<endl;

          unique_tangents.resize(unique_ordered_normals.size() + 1);
          if (dim == 3)
            unique_tangents[0] = cross_product_3d(
              unique_ordered_normals[0],
              unique_ordered_normals[unique_ordered_normals.size() - 1]);
          if (dim == 2)
            unique_tangents[0] = cross_product_2d(unique_ordered_normals[0]);
          for (unsigned int k = 1; k < unique_ordered_normals.size(); ++k)
            if (dim == 3)
              unique_tangents[k] =
                cross_product_3d(unique_ordered_normals[k],
                                 unique_ordered_normals[k - 1]);
            else if (dim == 2)
              unique_tangents[k] = cross_product_2d(unique_ordered_normals[k]);

          unique_tangents[unique_ordered_normals.size()] = unique_tangents[0];
          // cout<<i<<"  -->Unique tangents: "<<unique_tangents.size()<<endl;
          for (unsigned int k = 0; k < unique_tangents.size(); ++k)
            {
              unique_tangents[k] /= unique_tangents[k].norm();
              // cout<<unique_tangents[k]<<endl;
            }
          double geom_alpha = 0.5;

          geom_alpha = 2 * numbers::PI;
          if (unique_ordered_normals.size() > 1)
            {
              for (unsigned int k = 1; k < unique_ordered_normals.size(); ++k)
                geom_alpha -= acos(unique_ordered_normals[k] *
                                   unique_ordered_normals[k - 1]);
              geom_alpha -=
                acos(unique_ordered_normals[unique_ordered_normals.size() - 1] *
                     unique_ordered_normals[0]);
            }
          geom_alpha /= 4 * numbers::PI;
          hyp_alpha(i) = geom_alpha;

          Tensor<2, dim> C_matrix;
          for (unsigned int d = 0; d < dim; ++d)
            C_matrix[d][d] += geom_alpha;
          if (unique_ordered_normals.size() > 1)
            {
              for (unsigned int k = 0; k < unique_ordered_normals.size(); ++k)
                if (dim == 3)
                  {
                    C_matrix -=
                      1. / 4. / numbers::PI *
                      outer_product(cross_product_3d(unique_tangents[k + 1] -
                                                       unique_tangents[k],
                                                     unique_ordered_normals[k]),
                                    unique_ordered_normals[k]);
                    // cout<<"unique_tangents[k+1]:
                    // "<<unique_tangents[k+1]<<endl; cout<<"unique_tangents[k]:
                    // "<<unique_tangents[k]<<endl;
                    // cout<<"unique_ordered_normals[k]"<<unique_ordered_normals[k]<<endl;
                  }
                else if (dim == 2)
                  C_matrix -= 1 / 2 / numbers::PI *
                              outer_product(unique_tangents[k],
                                            unique_ordered_normals[k]);
            }
          // pcout<<"C_matrix: "<<C_matrix<<endl;
          for (unsigned int di = 0; di < dim; ++di)
            for (unsigned int dj = 0; dj < dim; ++dj)
              {
                C_ij[di * dim + dj][i] = C_matrix[di][dj];
                // pcout<<C_ij[di*dim+dj][i]<<std::endl;
              }
        }
    }

  pcout << "Done computing C_ij tensor" << endl;
  TrilinosWrappers::MPI::Vector error(hyp_alpha);
  error.sadd(-1.0, alpha);
  pcout << "Alpha abs error: " << error.l2_norm() << endl;
  pcout << "Alpha rel error: " << error.l2_norm() / alpha.l2_norm() << endl;

  // Calculating second free term as in Mantic et al. paper
  // Existence and evaluation of the two free terms in the hypersingular
  // boundary integral equation of potential theory
  // V Mantič, F Paris - Engineering Analysis with Boundary Elements, 1995
  // At each DOF, the free coefficient is a Tensor<1,dim>. It is null on dofs
  // located on smooth surfaces and becomes not null for non-smooth geometries.
  // We compute it here by means of integrals carried out in the cell parametric
  // plane. Because these are similar to those computed to obtain the
  // hypersingular quadrature, we implemented a method for such a computation in
  // the SingularKernelIntegral class. Cell by cell, and DOF by DOF, we compute
  // the portion of integral that gives the contribution of each cell to the
  // free coefficient of its DOFS
  pcout << "Computing vector b_i" << std::endl;
  std::vector<Tensor<1, dim>> free_term_b_all(dh.n_dofs());

  // we initialize an FEValues
  // object with the quadrature
  // formula for the integration of
  // the kernel in non singular
  // cells. This quadrature is
  // selected with the parameter
  // file, and needs to be quite
  // precise, since the functions we
  // are integrating are not
  // polynomial functions.
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int local_id = 0; local_id < fe->dofs_per_cell; ++local_id)
        {
          int global_id = local_dof_indices[local_id];
          if (this_cpu_set.is_element(global_id))
            {
              Assert(
                (*fe).has_support_points(),
                ExcMessage(
                  "The FE selected has no support points. This is not supported."));
              Point<dim - 1> P = (*fe).unit_support_point(local_id);

              SingularKernelIntegral<dim> singular_kernel_integrator(cell,
                                                                     *fe,
                                                                     *mapping,
                                                                     P);
              Tensor<1, dim>              b =
                singular_kernel_integrator.evaluate_free_term_b();
              std::set<types::global_dof_index> doubles =
                double_nodes_set[global_id];
              for (std::set<types::global_dof_index>::iterator it =
                     doubles.begin();
                   it != doubles.end();
                   it++)
                for (unsigned int d = 0; d < dim; ++d)
                  b_i[d][*it] += b[d];
            }
        }
    }
  for (unsigned int d = 0; d < dim; ++d)
    b_i[d].compress(VectorOperation::add);

  pcout << "Done computing vector b_i" << std::endl;

  pcout << "done computing free cefficients for hypersingular BIE" << std::endl;
}

template <int dim>
void
BEMProblem<dim>::compute_alpha()
{
  static TrilinosWrappers::MPI::Vector ones, zeros, dummy;
  if (ones.size() != dh.n_dofs())
    {
      ones.reinit(this_cpu_set, mpi_communicator);
      vector_shift(ones, -1.);
      zeros.reinit(this_cpu_set, mpi_communicator);
      dummy.reinit(this_cpu_set, mpi_communicator);
    }


  if (solution_method == "Direct")
    {
      neumann_matrix.vmult(alpha, ones);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_multipole_expansions(ones, zeros);
      fma.multipole_matr_vect_products(ones, zeros, alpha, dummy);
    }

  // alpha.print(pcout);
  // for (unsigned int i=0; i<alpha.size(); ++i)
  //    {
  //    cout<<std::setprecision(20)<<alpha(i)<<endl;
  //    }
}

template <int dim>
void
BEMProblem<dim>::vmult(TrilinosWrappers::MPI::Vector       &dst,
                       const TrilinosWrappers::MPI::Vector &src) const
{
  serv_phi = src;
  if (!have_dirichlet_bc)
    {
      vector_shift(serv_phi, -serv_phi.l2_norm());
    }
  serv_dphi_dn = src;



  TrilinosWrappers::MPI::Vector matrVectProdN;
  TrilinosWrappers::MPI::Vector matrVectProdD;

  matrVectProdN.reinit(this_cpu_set, mpi_communicator);
  matrVectProdD.reinit(this_cpu_set, mpi_communicator);

  dst = 0;


  serv_phi.scale(neumann_nodes);
  serv_dphi_dn.scale(dirichlet_nodes);

  if (solution_method == "Direct")
    {
      dirichlet_matrix.vmult(dst, serv_dphi_dn);
      dst *= -1;
      neumann_matrix.vmult_add(dst, serv_phi);
      serv_phi.scale(alpha);
      dst += serv_phi;
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_multipole_expansions(serv_phi, serv_dphi_dn);
      fma.multipole_matr_vect_products(serv_phi,
                                       serv_dphi_dn,
                                       matrVectProdN,
                                       matrVectProdD);
      serv_phi.scale(alpha);
      dst += matrVectProdD;
      dst *= -1;
      dst += matrVectProdN;
      dst += serv_phi;
    }

  // std::cout<<"*** "<<serv_phi(0)<<" or "<<serv_dphi_dn(0)<<"   src:
  // "<<src(0)<<"  dst: "<<dst(0)<<std::endl;
  // in fully neumann bc case, we have to rescale the vector to have a zero mean
  // one
  if (!have_dirichlet_bc)
    vector_shift(dst, -dst.l2_norm());
  dst.compress(VectorOperation::add);
}


template <int dim>
void
BEMProblem<dim>::compute_rhs(TrilinosWrappers::MPI::Vector       &dst,
                             const TrilinosWrappers::MPI::Vector &src) const
{
  serv_phi     = src;
  serv_dphi_dn = src;

  static TrilinosWrappers::MPI::Vector matrVectProdN;
  static TrilinosWrappers::MPI::Vector matrVectProdD;


  matrVectProdN.reinit(this_cpu_set, mpi_communicator);
  matrVectProdD.reinit(this_cpu_set, mpi_communicator);


  serv_phi.scale(dirichlet_nodes);
  serv_dphi_dn.scale(neumann_nodes);

  if (solution_method == "Direct")
    {
      neumann_matrix.vmult(dst, serv_phi);
      serv_phi.scale(alpha);
      dst += serv_phi;
      dst *= -1;
      dirichlet_matrix.vmult_add(dst, serv_dphi_dn);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_multipole_expansions(serv_phi, serv_dphi_dn);
      fma.multipole_matr_vect_products(serv_phi,
                                       serv_dphi_dn,
                                       matrVectProdN,
                                       matrVectProdD);
      serv_phi.scale(alpha);
      dst += matrVectProdN;
      dst += serv_phi;
      dst *= -1;
      dst += matrVectProdD;
    }
}



// @sect4{BEMProblem::solve_system}

// The next function simply solves
// the linear system.
template <int dim>
void
BEMProblem<dim>::solve_system(TrilinosWrappers::MPI::Vector       &phi,
                              TrilinosWrappers::MPI::Vector       &dphi_dn,
                              const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  Teuchos::TimeMonitor                       LocalTimer(*LacSolveTime);
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(100));

  system_rhs = 0;
  sol        = 0;
  alpha      = 0;


  compute_alpha();
  compute_hypersingular_free_coeffs();

  //   for (unsigned int i = 0; i < alpha.size(); i++)
  //      if (this_cpu_set.is_element(i))
  //         pcout<<std::setprecision(20)<<alpha(i)<<std::endl;



  compute_rhs(system_rhs, tmp_rhs);


  compute_constraints(constr_cpu_set, constraints, tmp_rhs);
  ConstrainedOperator<TrilinosWrappers::MPI::Vector, BEMProblem<dim>> cc(
    *this, constraints, constr_cpu_set, mpi_communicator);


  cc.distribute_rhs(system_rhs);
  system_rhs.compress(VectorOperation::insert);
  // vmult(sol,system_rhs);
  // Assert(sol.vector_partitioner().SameAs(system_rhs.vector_partitioner()),ExcMessage("Schizofrenia???"));
  // cc.vmult(sol,system_rhs);
  // Assert(sol.locally_owned_elements()==system_rhs.locally_owned_elements(),ExcMessage("IndexSet
  // a muzzo..."));
  // Assert(sol.vector_partitioner().SameAs(system_rhs.vector_partitioner()),ExcMessage("Ma
  // boh..."));


  if (solution_method == "Direct")
    {
      // SparseDirectUMFPACK &inv = fma.FMA_preconditioner(alpha);
      // solver.solve (*this, sol, system_rhs, inv);
      assemble_preconditioner();
      // solver.solve (cc, sol, system_rhs, PreconditionIdentity());
      sol.sadd(1., 0., system_rhs);
      solver.solve(cc, sol, system_rhs, preconditioner);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      TrilinosWrappers::PreconditionILU &fma_preconditioner =
        fma.FMA_preconditioner(alpha, constraints);
      solver.solve(cc, sol, system_rhs, fma_preconditioner);
      // solver.solve (cc, sol, system_rhs, PreconditionIdentity());
    }

  // cc.apply_constraint(sol);
  // pcout<<"sol = [";
  // for (unsigned int i = 0; i < dh.n_dofs(); i++)
  //    pcout<<sol(i)<<"; ";
  // pcout<<"];"<<std::endl;

  // for (unsigned int i = 0; i < sol.size(); i++)
  //   if (this_cpu_set.is_element(i))
  //      pcout<<std::setprecision(20)<<sol(i)<<std::endl;



  ///////////////////////////////////
  /*
    std::vector<Point<dim> > support_points(dh.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( mapping, dh,
    support_points); pcout<<"**solution "<<std::endl; for (unsigned int i = 0; i
    < alpha.size(); i++) if (this_cpu_set.is_element(i)) pcout<<i<<"
    ("<<this_mpi_process<<")
    "<<support_points[i](0)+support_points[i](1)+support_points[i](2)<<"
    "<<sol(i)<<std::endl;

     pcout<<"SOLUTION "<<std::endl;
     for (unsigned int i = 0; i < alpha.size(); i++)
         if (this_cpu_set.is_element(i))
            pcout<<i<<" ("<<this_mpi_process<<")  "<<sol(i)<<std::endl;
  */
  //////////////////////////////////

  for (types::global_dof_index i = 0; i < dirichlet_nodes.size(); i++)
    {
      if (this_cpu_set.is_element(i))
        {
          if (dirichlet_nodes(i) == 0)
            {
              phi(i) = sol(i);
            }
          else
            {
              dphi_dn(i) = sol(i);
            }
        }
    }
  phi(this_cpu_set.nth_index_in_set(0)) = phi(this_cpu_set.nth_index_in_set(0));
  dphi_dn(this_cpu_set.nth_index_in_set(0)) =
    dphi_dn(this_cpu_set.nth_index_in_set(0));
  phi.compress(VectorOperation::insert);
  dphi_dn.compress(VectorOperation::insert);

  // if (!have_dirichlet_bc)
  //   vector_shift(phi,-phi.l2_norm());

  // for (unsigned int i=0;i<dh.n_dofs();++i)
  // std::cout<<i<<" "<<tmp_rhs(i)<<" "<<dphi_dn(i)<<" "<<phi(i)<<"
  // "<<dirichlet_nodes(i)<<std::endl;

  // pcout<<"sol "<<std::endl;
  // for (unsigned int i = 0; i < sol.size(); i++)
  //    {
  // pcout<<i<<" "<<sol(i)<<" ";
  // std::set<unsigned int> doubles = double_nodes_set[i];
  // for (std::set<unsigned int>::iterator it = doubles.begin() ; it !=
  // doubles.end(); it++ )
  //    pcout<<*it<<"("<<dirichlet_nodes(*it)<<") ";
  // pcout<<"phi "<<phi(i)<<"  dphi_dn "<<dphi_dn(i);
  // pcout<<std::endl;
  //    }
}



// This method performs a Bem resolution,
// either in a direct or multipole method
template <int dim>
void
BEMProblem<dim>::solve(TrilinosWrappers::MPI::Vector       &phi,
                       TrilinosWrappers::MPI::Vector       &dphi_dn,
                       const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  if (solution_method == "Direct")
    {
      assemble_system();
      // neumann_matrix.print(std::cout);
      // dirichlet_matrix.print(std::cout);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_octree_blocking();
      // fma.compute_m2l_flags();
      fma.direct_integrals();
      fma.multipole_integrals();
    }

  solve_system(phi, dphi_dn, tmp_rhs);
}


template <int dim>
void
BEMProblem<dim>::compute_constraints(
  IndexSet                            &c_cpu_set,
  AffineConstraints<double>           &c,
  const TrilinosWrappers::MPI::Vector &tmp_rhs)

{
  Teuchos::TimeMonitor LocalTimer(*ConstraintsTime);
  // We need both the normal vector and surface gradients to apply correctly
  // dirichlet-dirichlet double node constraints. compute_normals();
  compute_surface_gradients(tmp_rhs);

  // communication is needed here: there is one matrix per process: thus the
  // vector needed to set inhomogeneities has to be copied locally
  Vector<double> localized_surface_gradients(vector_surface_gradients_solution);
  Vector<double> localized_normals(vector_normals_solution);
  Vector<double> localized_dirichlet_nodes(dirichlet_nodes);
  Vector<double> loc_tmp_rhs(tmp_rhs.size());
  loc_tmp_rhs = tmp_rhs;

  // we start clearing the constraint matrix
  c.clear();

  // here we prepare the constraint matrix so as to account for the presence
  // hanging nodes

  AffineConstraints<double> c_hn;
  DoFTools::make_hanging_node_constraints(dh, c_hn);
  c_hn.close();

  std::vector<types::subdomain_id> dofs_domain_association(dh.n_dofs());

  DoFTools::get_subdomain_association(dh, dofs_domain_association);
  // here we prepare the constraint matrix so as to account for the presence of
  // double and triple dofs

  // we start looping on the dofs
  for (types::global_dof_index i = 0; i < tmp_rhs.size(); i++)
    {
      // if (this_cpu_set.is_element(i))
      // {
      // in the next line we compute the "first" among the set of double nodes:
      // this node is the first dirichlet node in the set, and if no dirichlet
      // node is there, we get the first neumann node

      std::set<types::global_dof_index> doubles        = double_nodes_set[i];
      types::global_dof_index           firstOfDoubles = *doubles.begin();
      for (std::set<types::global_dof_index>::iterator it = doubles.begin();
           it != doubles.end();
           it++)
        {
          // if(this_cpu_set.is_element(*it))
          if (localized_dirichlet_nodes(*it) == 1)
            {
              firstOfDoubles = *it;
              break;
            }
        }

      // for each set of double nodes, we will perform the correction only once,
      // and precisely when the current node is the first of the set
      if (i == firstOfDoubles)
        {
          // the vector entry corresponding to the first node of the set does
          // not need modification, thus we erase ti form the set
          doubles.erase(i);

          // if the current (first) node is a dirichlet node, for all its
          // neumann doubles we will impose that the potential is equal to that
          // of the first node: this means that in the matrix vector product we
          // will put the potential value of the double node
          if (localized_dirichlet_nodes(i) == 1)
            {
              for (std::set<types::global_dof_index>::iterator it =
                     doubles.begin();
                   it != doubles.end();
                   it++)
                {
                  // if(this_cpu_set.is_element(*it))
                  {
                    if (localized_dirichlet_nodes(*it) == 1)
                      {
                        // this is the dirichlet-dirichlet case on flat edges:
                        // here we impose that dphi_dn on the two (or more)
                        // sides is equal.
                        double normal_distance = 0;

                        // types::global_dof_index owner_el_1 =
                        // DoFTools::count_dofs_with_subdomain_association (dh,
                        // dofs_domain_association[i]); types::global_dof_index
                        // owner_el_2 =
                        // DoFTools::count_dofs_with_subdomain_association (dh,
                        // dofs_domain_association[*it]);

                        for (unsigned int idim = 0; idim < dim; ++idim)
                          {
                            types::global_dof_index dummy_1 =
                              sub_wise_to_original[i];
                            types::global_dof_index dummy_2 =
                              sub_wise_to_original[*it];
                            types::global_dof_index index1 = vec_original_to_sub_wise
                              [gradient_dh.n_dofs() / dim * idim +
                               dummy_1]; // vector_start_per_process[dofs_domain_association[i]]
                                         // + idim * owner_el_1 + (i -
                                         // start_per_process[dofs_domain_association[i]]);
                                         // //gradient_dh.n_dofs()/dim*idim+i;//vector_start_per_process[this_mpi_process]
                                         // + (i -
                                         // start_per_process[this_mpi_process])
                                         // * dim + idim; //i*dim+idim
                            types::global_dof_index index2 = vec_original_to_sub_wise
                              [gradient_dh.n_dofs() / dim * idim +
                               dummy_2]; // vector_start_per_process[dofs_domain_association[*it]]
                                         // + idim * owner_el_2 + ((*it) -
                                         // start_per_process[dofs_domain_association[*it]]);//gradient_dh.n_dofs()/dim*idim+(*it);
                                         // //vector_start_per_process[this_mpi_process]
                                         // + ((*it) -
                                         // start_per_process[this_mpi_process])
                                         // * dim + idim;//(*it)*dim+idim
                            normal_distance += localized_normals[index1] *
                                               localized_normals[index2];
                          }
                        normal_distance /= normal_distance;
                        if (normal_distance < 1e-4)
                          {
                            c.add_line(*it);
                            c.add_entry(*it, i, 1);
                          }
                        // this is the dirichlet-dirichlet case on sharp edges:
                        // both normal gradients can be computed from surface
                        // gradients of phi and assingned as BC
                        else if (continuos_gradient)
                          {
                            c.add_line(*it);
                            double norm_i_norm_it = 0;
                            double surf_it_norm_i = 0;
                            double surf_i_norm_it = 0;

                            // types::global_dof_index owner_el_1 =
                            // DoFTools::count_dofs_with_subdomain_association
                            // (dh, dofs_domain_association[i]);
                            // types::global_dof_index owner_el_2 =
                            // DoFTools::count_dofs_with_subdomain_association
                            // (dh, dofs_domain_association[*it]);

                            // We no longer have a std::vector of Point<dim> so
                            // we need to perform the scalar product
                            for (unsigned int idim = 0; idim < dim; ++idim)
                              {
                                types::global_dof_index dummy_1 =
                                  sub_wise_to_original[i];
                                types::global_dof_index dummy_2 =
                                  sub_wise_to_original[*it];

                                types::global_dof_index index1 =
                                  vec_original_to_sub_wise
                                    [gradient_dh.n_dofs() / dim * idim +
                                     dummy_1]; // vector_start_per_process[dofs_domain_association[i]]
                                               // + idim * owner_el_1 + (i -
                                               // start_per_process[dofs_domain_association[i]]);//gradient_dh.n_dofs()/dim*idim+i;//vector_start_per_process[this_mpi_process]
                                               // + (i -
                                               // start_per_process[this_mpi_process])
                                               // * dim + idim;
                                types::global_dof_index index2 =
                                  vec_original_to_sub_wise
                                    [gradient_dh.n_dofs() / dim * idim +
                                     dummy_2]; // vector_start_per_process[dofs_domain_association[*it]]
                                               // + idim * owner_el_2 + ((*it) -
                                               // start_per_process[dofs_domain_association[*it]]);//gradient_dh.n_dofs()/dim*idim+(*it);//vector_start_per_process[this_mpi_process]
                                               // + ((*it) -
                                               // start_per_process[this_mpi_process])
                                               // * dim + idim;
                                norm_i_norm_it += localized_normals[index1] *
                                                  localized_normals[index2];
                                surf_it_norm_i +=
                                  localized_surface_gradients[index2] *
                                  localized_normals[index1];
                                surf_i_norm_it +=
                                  localized_surface_gradients[index1] *
                                  localized_normals[index2];
                              }
                            double this_normal_gradient =
                              (1.0 / (1.0 - pow(norm_i_norm_it, 2))) *
                              (surf_it_norm_i +
                               (surf_i_norm_it) * (norm_i_norm_it));
                            double other_normal_gradient =
                              (1.0 / (1.0 - pow(norm_i_norm_it, 2))) *
                              (surf_i_norm_it +
                               (surf_it_norm_i) * (norm_i_norm_it));
                            // std::cout<<"i="<<i<<" j="<<*it<<std::endl;
                            // std::cout<<"ni=("<<node_normals[i]<<")
                            // nj=("<<node_normals[*it]<<")"<<std::endl;
                            // std::cout<<"grad_s_phi_i=("<<node_surface_gradients[i]<<")
                            // grad_s_phi_j=("<<node_surface_gradients[*it]<<")"<<std::endl;
                            // std::cout<<"dphi_dn_i="<<this_normal_gradient<<"
                            // dphi_dn_j="<<other_normal_gradient<<std::endl;
                            // Point<3> this_full_gradient =
                            // node_normals[i]*this_normal_gradient +
                            // node_surface_gradients[i]; Point<3>
                            // other_full_gradient =
                            // node_normals[*it]*other_normal_gradient +
                            // node_surface_gradients[*it];
                            // std::cout<<"grad_phi_i=("<<this_full_gradient<<")
                            // grad_phi_j=("<<other_full_gradient<<")"<<std::endl;
                            c.add_line(i);
                            c.set_inhomogeneity(i, this_normal_gradient);
                            c.add_line(*it);
                            c.set_inhomogeneity(*it, other_normal_gradient);
                          }
                      }
                    else
                      {
                        c.add_line(*it);
                        c.set_inhomogeneity(*it, loc_tmp_rhs(i));
                        // dst(*it) = phi(*it)/alpha(*it);
                      }
                  }
                }
            }

          // if the current (first) node is a neumann node, for all its doubles
          // we will impose that the potential is equal to that of the first
          // node: this means that in the matrix vector product we will put the
          // difference between the potential at the fist node in the doubles
          // set, and the current double node
          if (localized_dirichlet_nodes(i) == 0)
            {
              for (std::set<types::global_dof_index>::iterator it =
                     doubles.begin();
                   it != doubles.end();
                   it++)
                {
                  c.add_line(*it);
                  c.add_entry(*it, i, 1);
                  // dst(*it) = phi(*it)/alpha(*it)-phi(i)/alpha(i);
                }
            }
        }
      // else if(firstOfDoubles == *doubles.begin())
      // {
      //   for(std::set<types::global_dof_index>::iterator it = doubles.begin()
      //   ; it != doubles.end(); it++ )
      //     if(*it!=firstOfDoubles)
      //       {
      //         c.add_line(*it);
      //         c.add_entry(*it,firstOfDoubles,1);
      //       }
      // }
      // }
    }

  c.merge(c_hn);
  c.close();

  c_cpu_set.clear();
  c_cpu_set.set_size(this_cpu_set.size());
  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    {
      if (this_cpu_set.is_element(i))
        {
          c_cpu_set.add_index(i);
          if (c.is_constrained(i))
            {
              const std::vector<std::pair<types::global_dof_index, double>>
                *entries = c.get_constraint_entries(i);
              for (types::global_dof_index j = 0; j < entries->size(); ++j)
                c_cpu_set.add_index((*entries)[j].first);
            }
        }
    }
  c_cpu_set.compress();

  /*
  pcout<<"CONSTAINT MATRIX CHECK "<<std::endl;
    for (types::global_dof_index i=0; i<dh.n_dofs(); ++i)
        {
        std::set <types::global_dof_index> duplicates = double_nodes_set[i];
        if (duplicates.size()>1)
           {
           pcout<<"Proc: "<<this_mpi_process<<" i= "<<i<<"
  ("<<localized_dirichlet_nodes(i)<<") duplicates: "; for
  (std::set<types::global_dof_index>::iterator pos = duplicates.begin(); pos
  !=duplicates.end(); pos++) pcout<<" "<<*pos; pcout<<std::endl;
           }
        }

    for(unsigned int i=0; i<dh.n_dofs(); ++i)
      if( (constraints.is_constrained(i)) )
        {pcout<<"Proc: "<<this_mpi_process<<" i= "<<i<<" (";
    const std::vector< std::pair < types::global_dof_index, double > >
      * entries = constraints.get_constraint_entries (i);
          pcout<<entries->size()<<")  Entries:";
    for(unsigned int j=0; j< entries->size(); ++j)
       pcout<<" "<<(*entries)[j].first<<" ("<<(*entries)[j].second<<") ";
         pcout<<" Inomogeneities:
  "<<constraints.get_inhomogeneity(i)<<std::endl;
        }
  */
}

template <int dim>
void
BEMProblem<dim>::assemble_preconditioner()
{
  if (is_preconditioner_initialized == false)
    {
      // pcout<<"Initialising preconditioner"<<std::endl;
      for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
        if (this_cpu_set.is_element(i))
          {
            // types::global_dof_index start_helper, end_helper;
            // if(i>preconditioner_band/2)
            //   start_helper = i-preconditioner_band/2;
            // else
            //   start_helper = (types::global_dof_index) 0;
            // if(i+preconditioner_band/2 < dh.n_dofs())
            //   end_helper = i+preconditioner_band/2;
            // else
            //   end_helper = dh.n_dofs();
            //   for(types::global_dof_index j=start_helper; j<end_helper; ++j)
            // pcout<<start_helper<<"
            // "<<std::min((types::global_dof_index)(i+preconditioner_band/2),(types::global_dof_index)dh.n_dofs())<<std::endl;
            types::global_dof_index start_helper =
              ((i) > preconditioner_band / 2) ? (i - preconditioner_band / 2) :
                                                ((types::global_dof_index)0);
            for (types::global_dof_index j = start_helper;
                 j < std::min((types::global_dof_index)(
                                i + preconditioner_band / 2),
                              (types::global_dof_index)dh.n_dofs());
                 ++j)
              preconditioner_sparsity_pattern.add(i, j);
          }
      preconditioner_sparsity_pattern.compress();
      band_system.reinit(preconditioner_sparsity_pattern);
      is_preconditioner_initialized = true;
    }
  else
    band_system = 0;


  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    {
      if (this_cpu_set.is_element(i))
        {
          if (constraints.is_constrained(i))
            band_system.add(i, i, 1);
          // types::global_dof_index start_helper, end_helper;
          // if(i>preconditioner_band/2)
          //   start_helper = i-preconditioner_band/2;
          // else
          //   start_helper = (types::global_dof_index) 0;
          // if(i+preconditioner_band/2 < dh.n_dofs())
          //   end_helper = i+preconditioner_band/2;
          // else
          //   end_helper = dh.n_dofs();
          // for(types::global_dof_index j=start_helper; j<end_helper; ++j)
          types::global_dof_index start_helper =
            ((i) > preconditioner_band / 2) ? (i - preconditioner_band / 2) :
                                              ((types::global_dof_index)0);

          for (types::global_dof_index j = start_helper;
               j <
               std::min((types::global_dof_index)i + preconditioner_band / 2,
                        (types::global_dof_index)dh.n_dofs());
               ++j)
            {
              if (constraints.is_constrained(i) == false)
                {
                  if (dirichlet_nodes(i) == 0)
                    {
                      // Nodo di Dirichlet
                      band_system.add(i, j, neumann_matrix(i, j));

                      if (i == j)
                        band_system.add(i, j, alpha(i));
                    }
                  else
                    band_system.add(i, j, -dirichlet_matrix(i, j));
                }
            }
        }
    }



  preconditioner.initialize(band_system);

  /*
  band_system.vmult(sol,alpha);
  pcout<<"**solution "<<std::endl;
   for (unsigned int i = 0; i < alpha.size(); i++)
       if (this_cpu_set.is_element(i))
          pcout<<i<<" ("<<this_mpi_process<<")  "<<sol(i)<<"
  "<<sol(i)<<std::endl;
  */
}



template <int dim>
void
BEMProblem<dim>::compute_gradients(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn)
{
  Teuchos::TimeMonitor LocalTimer(*GradientTime);

  // We need the solution to be stored on a parallel vector with ghost elements.
  // We let Trilinos take care of it.

  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi.reinit(glob_phi, false, true);
  TrilinosWrappers::MPI::Vector dphi_dn(ghosted_set);
  dphi_dn.reinit(glob_dphi_dn, false, true);



  // We reinit the gradient solution
  vector_gradients_solution.reinit(vector_this_cpu_set, mpi_communicator);

  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;


  // The matrix and rhs of our problem. We must decide if compute the mass
  // matrix just once and for all or not.
  TrilinosWrappers::SparseMatrix vector_gradients_matrix;
  TrilinosWrappers::MPI::Vector  vector_gradients_rhs(vector_this_cpu_set,
                                                     mpi_communicator);
  vector_gradients_matrix.reinit(vector_sparsity_pattern);


  // The vector FEValues to used in the assemblage
  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  // The scalar FEValues to interpolate the known value of phi
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_gradients |
                                update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  const unsigned int vector_n_q_points    = vector_fe_v.n_quadrature_points;
  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;
  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);


  std::vector<Tensor<1, dim>> phi_surf_grads(vector_n_q_points);
  std::vector<double>         phi_norm_grads(vector_n_q_points);
  std::vector<Vector<double>> q_vector_normals_solution(vector_n_q_points,
                                                        Vector<double>(dim));

  FullMatrix<double> local_gradients_matrix(vector_dofs_per_cell,
                                            vector_dofs_per_cell);
  Vector<double>     local_gradients_rhs(vector_dofs_per_cell);



  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  Quadrature<dim - 1>    dummy_quadrature(fe->get_unit_support_points());
  FEValues<dim - 1, dim> dummy_fe_v(*mapping,
                                    *fe,
                                    dummy_quadrature,
                                    update_values | update_gradients |
                                      update_normal_vectors |
                                      update_quadrature_points);

  const unsigned int                   dofs_per_cell = fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const unsigned int          n_q_points = dummy_fe_v.n_quadrature_points;
  std::vector<Tensor<1, dim>> dummy_phi_surf_grads(n_q_points);

  cell_it vector_cell = gradient_dh.begin_active();

  cell_it cell = dh.begin_active(), endc = dh.end();


  for (; cell != endc; ++cell, ++vector_cell)
    {
      Assert(cell->index() == vector_cell->index(), ExcInternalError());
      Assert(cell->subdomain_id() == vector_cell->subdomain_id(),
             ExcInternalError());

      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_v.reinit(cell);
          vector_fe_v.reinit(vector_cell);
          local_gradients_matrix = 0;
          local_gradients_rhs    = 0;
          const std::vector<Tensor<1, dim>> &vector_node_normals =
            vector_fe_v.get_normal_vectors();
          fe_v.get_function_gradients(phi, phi_surf_grads);
          fe_v.get_function_values(dphi_dn, phi_norm_grads);
          unsigned int comp_i, comp_j;



          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            {
              Tensor<1, dim> node_normal_grad_dir;
              for (unsigned int i = 0; i < dim; ++i)
                node_normal_grad_dir[i] = q_vector_normals_solution[q][i];
              Tensor<1, dim> gradient =
                vector_node_normals[q] * phi_norm_grads[q] + phi_surf_grads[q];
              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  comp_i = gradient_fe->system_to_component_index(i).first;
                  for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
                    {
                      comp_j = gradient_fe->system_to_component_index(j).first;
                      if (comp_i == comp_j)
                        {
                          local_gradients_matrix(i, j) +=
                            vector_fe_v.shape_value(i, q) *
                            vector_fe_v.shape_value(j, q) * vector_fe_v.JxW(q);
                        }
                    }
                  local_gradients_rhs(i) += (vector_fe_v.shape_value(i, q)) *
                                            gradient[comp_i] *
                                            vector_fe_v.JxW(q);
                }
            }
          vector_cell->get_dof_indices(vector_local_dof_indices);

          vector_constraints.distribute_local_to_global(
            local_gradients_matrix,
            local_gradients_rhs,
            vector_local_dof_indices,
            vector_gradients_matrix,
            vector_gradients_rhs);
        }
    }

  // At this point we can compress anything and solve via GMRES.
  vector_gradients_matrix.compress(VectorOperation::add);
  vector_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));

  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_gradients_matrix);
  solver.solve(vector_gradients_matrix,
               vector_gradients_solution,
               vector_gradients_rhs,
               mass_prec);

  vector_constraints.distribute(vector_gradients_solution);
}

template <int dim>
void
BEMProblem<dim>::compute_surface_gradients(
  const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  Teuchos::TimeMonitor          LocalTimer(*SurfaceGradientTime);
  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi.reinit(tmp_rhs, false, true);

  vector_surface_gradients_solution.reinit(vector_this_cpu_set,
                                           mpi_communicator);


  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;


  TrilinosWrappers::SparseMatrix vector_surface_gradients_matrix;
  TrilinosWrappers::MPI::Vector  vector_surface_gradients_rhs(
    vector_this_cpu_set, mpi_communicator);


  vector_surface_gradients_matrix.reinit(vector_sparsity_pattern);



  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_gradients |
                                update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  const unsigned int vector_n_q_points    = vector_fe_v.n_quadrature_points;
  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;
  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);

  std::vector<Tensor<1, dim>> phi_surf_grads(vector_n_q_points);
  std::vector<double>         phi_norm_grads(vector_n_q_points);
  std::vector<Vector<double>> q_vector_normals_solution(vector_n_q_points,
                                                        Vector<double>(dim));

  FullMatrix<double> local_gradients_matrix(vector_dofs_per_cell,
                                            vector_dofs_per_cell);
  Vector<double>     local_gradients_rhs(vector_dofs_per_cell);



  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  Quadrature<dim - 1>    dummy_quadrature(fe->get_unit_support_points());
  FEValues<dim - 1, dim> dummy_fe_v(*mapping,
                                    *fe,
                                    dummy_quadrature,
                                    update_values | update_gradients |
                                      update_normal_vectors |
                                      update_quadrature_points);

  const unsigned int                   dofs_per_cell = fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const unsigned int          n_q_points = dummy_fe_v.n_quadrature_points;
  std::vector<Tensor<1, dim>> dummy_phi_surf_grads(n_q_points);

  cell_it vector_cell = gradient_dh.begin_active();

  cell_it cell = dh.begin_active(), endc = dh.end();


  for (; cell != endc; ++cell, ++vector_cell)
    {
      Assert(cell->index() == vector_cell->index(), ExcInternalError());
      Assert(cell->subdomain_id() == vector_cell->subdomain_id(),
             ExcInternalError());

      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_v.reinit(cell);
          vector_fe_v.reinit(vector_cell);
          local_gradients_matrix = 0;
          local_gradients_rhs    = 0;
          fe_v.get_function_gradients(phi, phi_surf_grads);
          unsigned int comp_i, comp_j;



          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            {
              Tensor<1, dim> gradient = phi_surf_grads[q];
              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  comp_i = gradient_fe->system_to_component_index(i).first;
                  for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
                    {
                      comp_j = gradient_fe->system_to_component_index(j).first;
                      if (comp_i == comp_j)
                        {
                          local_gradients_matrix(i, j) +=
                            vector_fe_v.shape_value(i, q) *
                            vector_fe_v.shape_value(j, q) * vector_fe_v.JxW(q);
                        }
                    }
                  local_gradients_rhs(i) += (vector_fe_v.shape_value(i, q)) *
                                            gradient[comp_i] *
                                            vector_fe_v.JxW(q);
                }
            }
          vector_cell->get_dof_indices(vector_local_dof_indices);

          vector_constraints.distribute_local_to_global(
            local_gradients_matrix,
            local_gradients_rhs,
            vector_local_dof_indices,
            vector_surface_gradients_matrix,
            vector_surface_gradients_rhs);
        }
    }

  vector_surface_gradients_matrix.compress(VectorOperation::add);
  vector_surface_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));

  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_surface_gradients_matrix);

  solver.solve(vector_surface_gradients_matrix,
               vector_surface_gradients_solution,
               vector_surface_gradients_rhs,
               mass_prec);

  vector_constraints.distribute(vector_surface_gradients_solution);
}

template <int dim>
void
BEMProblem<dim>::compute_gradients_hypersingular(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn)
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  pcout << "Computing gradients with hypersingular integrals" << std::endl;

  TrilinosWrappers::MPI::Vector vector_hyp_gradients_solution(
    vector_this_cpu_set, mpi_communicator);
  TrilinosWrappers::MPI::Vector vector_b_free_coeff(vector_this_cpu_set,
                                                    mpi_communicator);

  Vector<double> phi_local(glob_phi);
  Vector<double> dphi_dn_local(glob_dphi_dn);

  // Tensor<1, dim> node_gradient;

  // Next, we initialize an FEValues
  // object with the quadrature
  // formula for the integration of
  // the kernel in non singular
  // cells. This quadrature is
  // selected with the parameter
  // file, and needs to be quite
  // precise, since the functions we
  // are integrating are not
  // polynomial functions.
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  const unsigned int n_q_points = fe_v.n_quadrature_points;

  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  pcout << fe->dofs_per_cell << " " << std::endl;

  // Now that we have checked that
  // the number of vertices is equal
  // to the number of degrees of
  // freedom, we construct a vector
  // of support points which will be
  // used in the local integrations:
  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);


  // After doing so, we can start the
  // integration loop over all cells,
  // where we first initialize the
  // FEValues object and get the
  // values of $\mathbf{\tilde v}$ at
  // the quadrature points (this
  // vector field should be constant,
  // but it doesn't hurt to be more
  // general):


  cell_it cell = dh.begin_active(), endc = dh.end();

  Tensor<1, dim> D;
  // Tensor<1, dim> R;
  Tensor<2, dim> H;
  double         s;

  Tensor<1, dim> integral;
  Tensor<1, dim> integral_2;
  // Tensor<1, dim> integral_3;
  integral[0]   = 0.0;
  integral[1]   = 0.0;
  integral[2]   = 0.0;
  integral_2[0] = 0.0;
  integral_2[1] = 0.0;
  integral_2[2] = 0.0;
  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      const std::vector<Point<dim>> &q_points    = fe_v.get_quadrature_points();
      const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();

      // We then form the integral over
      // the current cell for all
      // degrees of freedom (note that
      // this includes degrees of
      // freedom not located on the
      // current cell, a deviation from
      // the usual finite element
      // integrals). The integral that
      // we need to perform is singular
      // if one of the local degrees of
      // freedom is the same as the
      // support point $i$. A the
      // beginning of the loop we
      // therefore check wether this is
      // the case, and we store which
      // one is the singular index:
      for (types::global_dof_index i = 0; i < dh.n_dofs();
           ++i) // these must now be the locally owned dofs. the rest should
                // stay the same
        {
          Tensor<1, dim> integral;
          Tensor<1, dim> b_integral;
          if (this_cpu_set.is_element(i))
            {
              bool         is_singular    = false;
              unsigned int singular_index = numbers::invalid_unsigned_int;

              for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                // if(local_dof_indices[j] == i)
                if (double_nodes_set[i].count(local_dof_indices[j]) > 0)
                  {
                    singular_index = j;
                    is_singular    = true;
                    break;
                  }

              // We then perform the
              // integral. If the index $i$
              // is not one of the local
              // degrees of freedom, we
              // simply have to add the
              // single layer terms to the
              // right hand side, and the
              // double layer terms to the
              // matrix:

              if (is_singular == false)
                {
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                      const Tensor<1, dim> R = q_points[q] - support_points[i];
                      LaplaceKernel::kernels(R, H, D, s);
                      for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                        {
                          integral += -phi_local(local_dof_indices[j]) *
                                        (H * normals[q]) *
                                        fe_v.shape_value(j, q) * fe_v.JxW(q) +
                                      dphi_dn_local(local_dof_indices[j]) * D *
                                        fe_v.shape_value(j, q) * fe_v.JxW(q);
                          b_integral += -1.0 * (H * normals[q]) *
                                        fe_v.shape_value(j, q) * fe_v.JxW(q);
                        }
                    }
                }
              else
                {
                  // Now we treat the more
                  // delicate case. If we
                  // are here, this means
                  // that the cell that
                  // runs on the $j$ index
                  // contains
                  // support_point[i]. In
                  // this case both the
                  // single and the double
                  // layer potential are
                  // singular, and they
                  // require special
                  // treatment.
                  //

                  Assert(
                    (*fe).has_support_points(),
                    ExcMessage(
                      "The FE selected has no support points. This is not supported."));
                  Point<dim - 1> P = (*fe).unit_support_point(singular_index);
                  // pcout<<"P: "<<P<<std::endl;

                  SingularKernelIntegral<dim> sing_kernel_integrator(cell,
                                                                     *fe,
                                                                     *mapping,
                                                                     P);
                  std::vector<Tensor<1, dim>> Vk_integrals =
                    sing_kernel_integrator.evaluate_VkNj_integrals();
                  std::vector<Tensor<1, dim>> Wk_integrals =
                    sing_kernel_integrator.evaluate_WkNj_integrals();
                  Tensor<1, dim> singular_cell_contribution_hyp;
                  Tensor<1, dim> singular_cell_contribution_str;
                  // Tensor<1,dim> b_singular_cell_contribution_hyp;
                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                    {
                      // const Tensor<1, dim> R =
                      // support_points[local_dof_indices[j]] -
                      // support_points[i]; pcout<<"* "<<cell<<"  "<<R<<"
                      // "<<support_points[local_dof_indices[j]]<<std::endl;

                      singular_cell_contribution_hyp +=
                        -phi_local(local_dof_indices[j]) * Vk_integrals[j];
                      singular_cell_contribution_str +=
                        dphi_dn_local(local_dof_indices[j]) * Wk_integrals[j];
                      // this was an attempt to compute b_i in an alternative,
                      // numerical way, as alpha. Couldn't get it to work
                      // b_singular_cell_contribution_hyp+= -Vk_integrals[j];

                      // pcout<<"*** "<<cell<<"
                      // "<<dphi_dn_local(local_dof_indices[j])<<"
                      // "<<Wk_integrals[j]<<std::endl; pcout<<"j "<<j<<"
                      // "<<cell<<"  "<<phi_local(local_dof_indices[j])<<"
                      // "<<Vk_integrals[j]<<std::endl;
                    }
                  // pcout<<cell<<"   "<<singular_cell_contribution_str<<"
                  // "<<singular_cell_contribution_hyp<<std::endl;
                  // integral_3+=singular_cell_contribution_str+singular_cell_contribution_hyp;
                  // pcout<<"Qmark Hyp: "<<integral_3<<std::endl;
                  integral += singular_cell_contribution_str +
                              singular_cell_contribution_hyp;
                  // b_integral+= b_singular_cell_contribution_hyp;
                }

              unsigned int scalar_dh_index = sub_wise_to_original[i];
              unsigned int vector_dh_index_x_component =
                vec_original_to_sub_wise[scalar_dh_index + 0 * dh.n_dofs()];
              unsigned int vector_dh_index_y_component =
                vec_original_to_sub_wise[scalar_dh_index + 1 * dh.n_dofs()];
              unsigned int vector_dh_index_z_component =
                vec_original_to_sub_wise[scalar_dh_index + 2 * dh.n_dofs()];
              vector_hyp_gradients_solution(vector_dh_index_x_component) +=
                integral[0];
              vector_hyp_gradients_solution(vector_dh_index_y_component) +=
                integral[1];
              vector_hyp_gradients_solution(vector_dh_index_z_component) +=
                integral[2];
              // vector_b_free_coeff(vector_dh_index_x_component)+=b_integral[0];
              // vector_b_free_coeff(vector_dh_index_y_component)+=b_integral[1];
              // vector_b_free_coeff(vector_dh_index_z_component)+=b_integral[2];
            }
        }
    }
  vector_hyp_gradients_solution.compress(VectorOperation::add);
  vector_b_free_coeff.compress(VectorOperation::add);


  //  for (types::global_dof_index i = 0; i < dh.n_dofs();
  //       ++i) // these must now be the locally owned dofs. the rest should
  //            // stay the same
  //    {
  //      if (this_cpu_set.is_element(i))
  //        { pcout<<i<<"->    Support point: "<<support_points[i]<<std::endl;
  //          pcout << free_term_b_all[i][0] << " "
  //                << free_term_b_all[i][1] << " "
  //                << free_term_b_all[i][2] << std::endl;
  //
  //        }
  //    }



  // we now have all the ingredients for the computation of the gradients
  // through the hypersingual BIE. For each DOF, such a BIE is a vector
  // equation, of which we already have the right hand side (assembled with the
  // integrals involving phi and dphi_dn). In the right hand side we also have
  // the potential multiplied by the second --- Mantic --- free coefficient As
  // for the left hand side, the gradient of phi (our unknown) multiplies tensor
  // C. So, for each line we must invert C to obtain our gradient
  for (types::global_dof_index i = 0; i < dh.n_dofs();
       ++i) // these must now be the locally owned dofs. the rest should
            // stay the same
    {
      if (this_cpu_set.is_element(i))
        {
          // these will be useful
          unsigned int scalar_dh_index = sub_wise_to_original[i];
          unsigned int vector_dh_index_x_component =
            vec_original_to_sub_wise[scalar_dh_index + 0 * dh.n_dofs()];
          unsigned int vector_dh_index_y_component =
            vec_original_to_sub_wise[scalar_dh_index + 1 * dh.n_dofs()];
          unsigned int vector_dh_index_z_component =
            vec_original_to_sub_wise[scalar_dh_index + 2 * dh.n_dofs()];
          // cout<<this_mpi_process<<"  Scalar index original
          // "<<scalar_dh_index<<"   Sub-wise correspondent "<<i<<"  Vector x
          // index original "<< scalar_dh_index+ 0+dh.n_dofs()<<"   Sub-wise
          // correspondent "<<vector_dh_index_x_component<<"  Vector y index
          // original "<< scalar_dh_index+ 1+dh.n_dofs()<<"   Sub-wise
          // correspondent "<<vector_dh_index_y_component<<"  Vector z index
          // original "<< scalar_dh_index+ 2+dh.n_dofs()<<"   Sub-wise
          // correspondent "<<vector_dh_index_z_component<<endl;
          // cout<<this_mpi_process<<"  Range:
          // ("<<vector_hyp_gradients_solution.local_range().first<<","<<vector_hyp_gradients_solution.local_range().second<<")
          // "<<vector_dh_index_x_component<<" "<<vector_dh_index_y_component<<"
          // "<<vector_dh_index_z_component<<endl; pcout<<i<<"->    Support
          // point: "<<support_points[i]<<std::endl; pcout<<"b(mantic):
          // "<<free_term_b_all[i]<<std::endl; pcout<<"b(alt):
          // "<<vector_b_free_coeff(i)<<" "
          //                  <<vector_b_free_coeff(i+dh.n_dofs())<<" "
          //                  <<vector_b_free_coeff(i+2*dh.n_dofs())<<std::endl;
          // pcout<<"phi: "<<phi_local[i]<<std::endl;
          //  we must first reassemble the free coefficient vector b
          Tensor<1, dim> b;
          for (unsigned int d = 0; d < dim; ++d)
            b[d] = b_i[d][i];

          // we also need to reassemble free coefficient tensor C_ij
          FullMatrix<double> C(dim, dim);
          FullMatrix<double> Cinv(dim, dim);
          for (unsigned int di = 0; di < dim; ++di)
            for (unsigned int dj = 0; dj < dim; ++dj)
              C(di, dj) = C_ij[di * dim + dj](i);

          // we will also need the inverse of C
          Tensor<2, dim> CC;
          // pcout<<"C: "<<std::endl;
          // C.print_formatted(std::cout, 7, true,10,"0");
          Cinv.invert(C);
          Cinv.copy_to(CC);
          // pcout<<"Cinv: "<<std::endl;
          // Cinv.print(std::cout, 5, 5);

          // now let's assemble the right hand side of the hypersingular BIE
          Tensor<1, dim> rhs;
          rhs[0] = vector_hyp_gradients_solution(vector_dh_index_x_component);
          rhs[1] = vector_hyp_gradients_solution(vector_dh_index_y_component);
          rhs[2] = vector_hyp_gradients_solution(vector_dh_index_z_component);

          rhs += -b * phi_local[i];
          // pcout<<rhs<<std::endl;

          // and finally multiply C^-1 by the rhs to obtain the gradient
          Tensor<1, dim> hyp_gradient;
          hyp_gradient = CC * rhs;
          vector_hyp_gradients_solution(vector_dh_index_x_component) =
            hyp_gradient[0];
          vector_hyp_gradients_solution(vector_dh_index_y_component) =
            hyp_gradient[1];
          vector_hyp_gradients_solution(vector_dh_index_z_component) =
            hyp_gradient[2];
          // pcout<<"Hyp. Rhs:"<<rhs<<std::endl;
          // pcout<<"Hyp. Gradient:"<<hyp_gradient<<std::endl;
        }
    }

  vector_hyp_gradients_solution.compress(VectorOperation::insert);
  vector_gradients_solution = vector_hyp_gradients_solution;
  pcout << "done computing gradients with hypersingular integrals" << std::endl;
}

template <int dim>
void
BEMProblem<dim>::compute_normals()
{
  Teuchos::TimeMonitor LocalTimer(*NormalsTime);
  vector_normals_solution.reinit(vector_this_cpu_set, mpi_communicator);

  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;


  TrilinosWrappers::SparseMatrix vector_normals_matrix;
  TrilinosWrappers::MPI::Vector  vector_normals_rhs(vector_this_cpu_set,
                                                   mpi_communicator);


  vector_normals_matrix.reinit(vector_sparsity_pattern);



  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  const unsigned int vector_n_q_points = vector_fe_v.n_quadrature_points;

  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;

  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);

  std::vector<Vector<double>> q_vector_normals_solution(vector_n_q_points,
                                                        Vector<double>(dim));

  FullMatrix<double> local_normals_matrix(vector_dofs_per_cell,
                                          vector_dofs_per_cell);
  Vector<double>     local_normals_rhs(vector_dofs_per_cell);


  cell_it vector_cell = gradient_dh.begin_active(),
          vector_endc = gradient_dh.end();


  for (; vector_cell != vector_endc; ++vector_cell)
    {
      if (vector_cell->subdomain_id() == this_mpi_process)
        {
          vector_fe_v.reinit(vector_cell);
          local_normals_matrix = 0;
          local_normals_rhs    = 0;
          const std::vector<Tensor<1, dim>> &vector_node_normals =
            vector_fe_v.get_normal_vectors();
          unsigned int comp_i, comp_j;

          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
              {
                comp_i = gradient_fe->system_to_component_index(i).first;
                for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
                  {
                    comp_j = gradient_fe->system_to_component_index(j).first;
                    if (comp_i == comp_j)
                      {
                        local_normals_matrix(i, j) +=
                          vector_fe_v.shape_value(i, q) *
                          vector_fe_v.shape_value(j, q) * vector_fe_v.JxW(q);
                      }
                  }

                local_normals_rhs(i) += (vector_fe_v.shape_value(i, q)) *
                                        vector_node_normals[q][comp_i] *
                                        vector_fe_v.JxW(q);
              }

          vector_cell->get_dof_indices(vector_local_dof_indices);

          vector_constraints.distribute_local_to_global(
            local_normals_matrix,
            local_normals_rhs,
            vector_local_dof_indices,
            vector_normals_matrix,
            vector_normals_rhs);
        }
    }

  vector_normals_matrix.compress(VectorOperation::add);
  vector_normals_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));
  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_normals_matrix);


  solver.solve(vector_normals_matrix,
               vector_normals_solution,
               vector_normals_rhs,
               mass_prec);

  vector_constraints.distribute(vector_normals_solution);
}

template <int dim>
void
BEMProblem<dim>::adaptive_refinement(
  const TrilinosWrappers::MPI::Vector &error_vector)
{
  Vector<float>  estimated_error_per_cell(comp_dom.tria.n_active_cells());
  Vector<double> helper(error_vector);

  KellyErrorEstimator<dim - 1, dim>::estimate(
    *mapping, dh, QGauss<dim - 2>(3), {}, helper, estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(comp_dom.tria,
                                                  estimated_error_per_cell,
                                                  refinement_threshold,
                                                  coarsening_threshold);
  comp_dom.tria.prepare_coarsening_and_refinement();
  comp_dom.tria.execute_coarsening_and_refinement();
}



template class BEMProblem<2>;
template class BEMProblem<3>;
