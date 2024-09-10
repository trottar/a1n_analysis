#ifndef MATH_DF_H
#define MATH_DF_H

#include <cmath>
#include "TMath.h"

namespace math_df {
   double GetMean(std::vector<double> x); 
   double GetStandardDeviation(std::vector<double> x,bool besselCor=false); 
   double GetVariance(std::vector<double> x,bool besselCor=false);
} 

#endif  
