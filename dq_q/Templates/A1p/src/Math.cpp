#include "../include/Math.h"
namespace math_df { 
   //______________________________________________________________________________
   double GetMean(std::vector<double> x){
      int N = x.size();
      double sum=0;
      for(int i=0;i<N;i++) sum += x[i];
      double mean = sum/( (double)N );
      return mean;
   }
   //______________________________________________________________________________
   double GetVariance(std::vector<double> x,bool besselCor){
      int N = x.size();
      double mean = GetMean(x);
      double sum=0;
      int den = N;
      if(besselCor){
	 den = N - 1;
      }
      for(int i=0;i<N;i++){
	 sum += pow(x[i]-mean,2);
      }
      double var   = sum/( (double)den );
      return var;
   }
   //______________________________________________________________________________
   double GetStandardDeviation(std::vector<double> x,bool besselCor){
      double var   = GetVariance(x,besselCor);
      double stdev = sqrt(var);
      return stdev;
   }
}
