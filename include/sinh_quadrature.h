#ifndef sinh_quadrature_h
#define sinh_quadrature_h

#include <deal.II/base/quadrature_lib.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace dealii;

namespace
{
  inline void
  make_sinh_points_1d(const unsigned int n,
                      const double       singularity,
                      const double       beta,
                      std::vector<double> &points,
                      std::vector<double> &weights)
  {
    const QGauss<1> base(n);
    const double    eps = 1e-12;
    const double    s   = std::max(0.0, std::min(1.0, singularity));

    points.clear();
    weights.clear();

    auto add_interval = [&](const double a,
                            const double b,
                            const bool   cluster_at_left) {
      if (b - a < eps)
        return;

      for (unsigned int q = 0; q < base.size(); ++q)
        {
          const double t = base.point(q)[0];
          const double w = base.weight(q);

          double x;
          double dx_dt;
          if (cluster_at_left)
            {
              x     = a + (b - a) * std::sinh(beta * t) / std::sinh(beta);
              dx_dt = (b - a) * beta * std::cosh(beta * t) / std::sinh(beta);
            }
          else
            {
              x = b - (b - a) * std::sinh(beta * (1.0 - t)) /
                        std::sinh(beta);
              dx_dt = (b - a) * beta * std::cosh(beta * (1.0 - t)) /
                      std::sinh(beta);
            }

          points.push_back(x);
          weights.push_back(w * dx_dt);
        }
    };

    if (s <= eps)
      add_interval(0.0, 1.0, true);
    else if (s >= 1.0 - eps)
      add_interval(0.0, 1.0, false);
    else
      {
        add_interval(0.0, s, false);
        add_interval(s, 1.0, true);
      }
  }
}

template <int dim>
class QSinh : public Quadrature<dim>
{
public:
  QSinh(const unsigned int n,
        const Point<dim>  &singularity,
        const double       beta = 3.0);
};

template <>
inline QSinh<1>::QSinh(const unsigned int n,
                       const Point<1>  &singularity,
                       const double     beta)
  : Quadrature<1>()
{
  std::vector<double> x;
  std::vector<double> wx;
  make_sinh_points_1d(n, singularity[0], beta, x, wx);

  std::vector<Point<1>> points(x.size());
  std::vector<double>   weights(x.size());
  for (unsigned int i = 0; i < x.size(); ++i)
    {
      points[i][0] = x[i];
      weights[i]   = wx[i];
    }

  this->quadrature_points = points;
  this->weights           = weights;
}

template <>
inline QSinh<2>::QSinh(const unsigned int n,
                       const Point<2>  &singularity,
                       const double     beta)
  : Quadrature<2>()
{
  std::vector<double> x;
  std::vector<double> wx;
  std::vector<double> y;
  std::vector<double> wy;
  make_sinh_points_1d(n, singularity[0], beta, x, wx);
  make_sinh_points_1d(n, singularity[1], beta, y, wy);

  std::vector<Point<2>> points;
  std::vector<double>   weights;
  points.reserve(x.size() * y.size());
  weights.reserve(x.size() * y.size());

  for (unsigned int i = 0; i < x.size(); ++i)
    for (unsigned int j = 0; j < y.size(); ++j)
      {
        Point<2> p;
        p[0] = x[i];
        p[1] = y[j];
        points.push_back(p);
        weights.push_back(wx[i] * wy[j]);
      }

  this->quadrature_points = points;
  this->weights           = weights;
}

#endif
