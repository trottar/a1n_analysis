#ifndef BBS_H 
#define BBS_H

// pQCD model, no higher-twist effects, evaluated at Q2 = 4 GeV^2 
// produces various observables as a function of x 
// - structure functions: F1p, g1p, F1n, g1n 
// - asymmetries: A1p, A1n
// - PDFs: u, d, s, and g (+ polarized) 
// references 
// - Int J. Mod Phys. A 13, 5573 (1998), arXiv:9708335 [hep-ph] 

#include <cstdlib>
#include <iostream>

#include "TMath.h"

class BBS {

   private:
      double fAlpha_q,fAlpha_g;
      double fAu,fAd,fAs,fAg; 
      double fBu,fBd,fBs,fBg; 
      double fCu,fCd,fCs; 
      double fDu,fDd,fDs;
      double fQu,fQd,fQs;
 
   public:
      BBS(); 
      ~BBS();

      double get_u_uBar(double x);  
      double get_d_dBar(double x);  
      double get_s_sBar(double x); 
      double get_g(double x); 
      double get_delta_u_uBar(double x);  
      double get_delta_d_dBar(double x);  
      double get_delta_s_sBar(double x);  
      double get_delta_g(double x); 

      double get_g1n(double x);  
      double get_F1n(double x);  
      double get_A1n(double x); 
      double get_g1p(double x);  
      double get_F1p(double x);  
      double get_A1p(double x);  

}; 

#endif 
