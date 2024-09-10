#ifndef AVAKIAN_H 
#define AVAKIAN_H

// pQCD model with orbital angular momentum 
// produces various observables as a function of x  
// - structure functions: F1p, g1p, F1n, g1n 
// - asymmetries: A1p, A1n
// - PDFs: u, d, s, and g (+ polarized) 
// references 
// - Phys. Rev. Lett. 99 082001 (1999)
// - Int J Mod Phys A 13, 5573 (1998) 
// - Nucl. Phys. B 441 197 (1995)  

#include <cstdlib>
#include <iostream>

#include "TMath.h"

class Avakian {

   private:
      double fAlpha_q,fAlpha_g;
      double fAu,fAd,fAs,fAg; 
      double fBu,fBd,fBs,fBg; 
      double fCu,fCd,fCs,fCu_pr,fCd_pr; 
      double fDu,fDd,fDs;
      double fQu,fQd,fQs;

      double get_u_plus(double x);  
      double get_d_plus(double x);  
      double get_s_plus(double x); 
      double get_u_minus(double x);  
      double get_d_minus(double x);  
      double get_s_minus(double x);  
 
   public:
      Avakian(); 
      ~Avakian();

      double get_delta_u(double x);  
      double get_delta_d(double x);  
      double get_delta_s(double x);  
      double get_delta_g(double x);  
      double get_u(double x);  
      double get_d(double x);  
      double get_s(double x); 
      double get_g(double x); 

      double get_g1n(double x);  
      double get_F1n(double x);  
      double get_A1n(double x); 
      double get_g1p(double x);  
      double get_F1p(double x);  
      double get_A1p(double x);  

}; 

#endif 
