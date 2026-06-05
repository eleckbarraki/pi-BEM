#ifndef subdivision_quadrature_h
#define subdivision_quadrature_h

#include <deal.II/base/quadrature_lib.h>

#include <algorithm>
#include <vector>

using namespace dealii;

namespace
{
  inline std::vector<std::pair<double, double>>
  split_interval_at_point(const double point)
  {
    const double eps = 1e-12;
    const double s   = std::max(0.0, std::min(1.0, point));

    std::vector<std::pair<double, double>> intervals;
    if (s > eps)
      intervals.emplace_back(0.0, s);
    if (s < 1.0 - eps)
      intervals.emplace_back(s, 1.0);

    if (intervals.empty())
      intervals.emplace_back(0.0, 1.0);

    return intervals;
  }
}

template <int dim>
class QSubdivision : public Quadrature<dim>
{
public:
  QSubdivision(const unsigned int n, const Point<dim> &split_point);
};

template <>
inline QSubdivision<1>::QSubdivision(const unsigned int n,
                                     const Point<1>  &split_point)
  : Quadrature<1>()
{
  const QGauss<1> base(n);
  const auto      intervals = split_interval_at_point(split_point[0]);

  std::vector<Point<1>> points;
  std::vector<double>   weights;

  for (const auto &interval : intervals)
    {
      const double a = interval.first;
      const double b = interval.second;

      for (unsigned int q = 0; q < base.size(); ++q)
        {
          Point<1> p;
          p[0] = a + (b - a) * base.point(q)[0];
          points.push_back(p);
          weights.push_back((b - a) * base.weight(q));
        }
    }

  this->quadrature_points = points;
  this->weights           = weights;
}

template <>
inline QSubdivision<2>::QSubdivision(const unsigned int n,
                                     const Point<2>  &split_point)
  : Quadrature<2>()
{
  const QGauss<1> base(n);
  const auto      x_intervals = split_interval_at_point(split_point[0]);
  const auto      y_intervals = split_interval_at_point(split_point[1]);

  std::vector<Point<2>> points;
  std::vector<double>   weights;
  points.reserve(x_intervals.size() * y_intervals.size() * n * n);
  weights.reserve(x_intervals.size() * y_intervals.size() * n * n);

  for (const auto &x_interval : x_intervals)
    for (const auto &y_interval : y_intervals)
      {
        const double ax = x_interval.first;
        const double bx = x_interval.second;
        const double ay = y_interval.first;
        const double by = y_interval.second;

        for (unsigned int i = 0; i < base.size(); ++i)
          for (unsigned int j = 0; j < base.size(); ++j)
            {
              Point<2> p;
              p[0] = ax + (bx - ax) * base.point(i)[0];
              p[1] = ay + (by - ay) * base.point(j)[0];
              points.push_back(p);
              weights.push_back((bx - ax) * (by - ay) *
                                base.weight(i) * base.weight(j));
            }
      }

  this->quadrature_points = points;
  this->weights           = weights;
}

#endif
