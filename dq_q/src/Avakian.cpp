#include "../include/Avakian.h"
//______________________________________________________________________________
Avakian::Avakian(){
   // quark charge 
   fQu      = 2./3.; 
   fQd      = -1./3.; 
   fQs      = -1./3.; 
   // from Int J Mod Phys A 13, 5573 (1998) 
   fAlpha_q = 1.313;  // ± 0.056 
   fAlpha_g = 1.233;  // ± 0.073 
   fAu      = 3.088;
   fAd      = 0.343;
   fAg      = 1.019;
   fBu      = -3.010; // ± 0.156
   fBd      = -0.265;
   fBg      = -0.339; // ± 0.454
   fCu      = 2.143;  // ± 0.137
   fCd      = 1.689;  // ± 0.227
   fDu      = -2.065; // ± 0.148
   fDd      = -1.610;
   fCs      = 0.334;
   fAs      = 0.001;
   fBs      = 0.041;
   fDs      = -0.292; // ± 0.042 
   // from Avakian's paper
   fCu_pr   = 0.493; // ± 0.249
   fCd_pr   = 1.592; // ± 0.378
}
//______________________________________________________________________________
Avakian::~Avakian(){

}
//______________________________________________________________________________
double Avakian::get_A1p(double x){
   double g1p = get_g1p(x);
   double F1p = get_F1p(x);
   double A1p = g1p/F1p;
   return A1p;
}
//______________________________________________________________________________
double Avakian::get_A1n(double x){
   double g1n = get_g1n(x);
   double F1n = get_F1n(x);
   double A1n = g1n/F1n;
   return A1n;
}
//______________________________________________________________________________
double Avakian::get_g1p(double x){
   double g1p = 0.5*( fQu*fQu*get_delta_u(x) + fQd*fQd*get_delta_d(x) + fQs*fQs*get_delta_s(x) ); 
   return g1p;
}
//______________________________________________________________________________
double Avakian::get_F1p(double x){
   double F1p = 0.5*( fQu*fQu*get_u(x) + fQd*fQd*get_d(x) + fQs*fQs*get_s(x) ); 
   return F1p;
}
//______________________________________________________________________________
double Avakian::get_g1n(double x){
   // NOTE the charges!
   double g1n = 0.5*( fQd*fQd*get_delta_u(x) + fQu*fQu*get_delta_d(x) + fQs*fQs*get_delta_s(x) ); 
   return g1n;
}
//______________________________________________________________________________
double Avakian::get_F1n(double x){
   // NOTE the charges!
   double F1n = 0.5*( fQd*fQd*get_u(x) + fQu*fQu*get_d(x) + fQs*fQs*get_s(x) ); 
   return F1n;
}
//______________________________________________________________________________
double Avakian::get_u(double x){
   double u = get_u_plus(x) + get_u_minus(x); 
   return u;
}
//______________________________________________________________________________
double Avakian::get_d(double x){
   double d = get_d_plus(x) + get_d_minus(x); 
   return d;
}
//______________________________________________________________________________
double Avakian::get_s(double x){
   double s = get_s_plus(x) + get_s_minus(x); 
   return s;
}
//______________________________________________________________________________
double Avakian::get_delta_u(double x){
   double delta_u = get_u_plus(x) - get_u_minus(x); 
   return delta_u;
}
//______________________________________________________________________________
double Avakian::get_delta_d(double x){
   double delta_d = get_d_plus(x) - get_d_minus(x); 
   return delta_d;
}
//______________________________________________________________________________
double Avakian::get_delta_s(double x){
   double delta_s = get_s_plus(x) - get_s_minus(x); 
   return delta_s;
}
//______________________________________________________________________________
double Avakian::get_g(double x){
   double xa = TMath::Power(x,fAlpha_g);
   double f  = (1./xa)*TMath::Power(1.-x,4.)*( fAg + fBg*(1.-x) )*( 1. + TMath::Power(1.-x,2.) );
   return f;
}
//______________________________________________________________________________
double Avakian::get_delta_g(double x){
   double xa = TMath::Power(x,fAlpha_g);
   double f  = (1./xa)*TMath::Power(1.-x,4.)*( fAg + fBg*(1.-x) )*( 1. - TMath::Power(1.-x,2.) );
   return f;
}
//______________________________________________________________________________
double Avakian::get_u_plus(double x){
   double xa    = TMath::Power(x,fAlpha_q); 
   double uplus = (1./xa)*( fAu*TMath::Power(1.-x,3.) + fBu*TMath::Power(1.-x,4.) ); 
   return uplus; 
}
//______________________________________________________________________________
double Avakian::get_u_minus(double x){
   double xa    = TMath::Power(x,fAlpha_q); 
   double uminus = (1./xa)*( fCu*TMath::Power(1.-x,5.) + fCu_pr*TMath::Power(1.-x,5.)*TMath::Power(TMath::Log(1-x),2.) 
                           + fDu*TMath::Power(1.-x,6.) ); 
   return uminus; 
}
//______________________________________________________________________________
double Avakian::get_d_plus(double x){
   double xa    = TMath::Power(x,fAlpha_q); 
   double dplus = (1./xa)*( fAd*TMath::Power(1.-x,3.) + fBd*TMath::Power(1.-x,4.) ); 
   return dplus; 
}
//______________________________________________________________________________
double Avakian::get_d_minus(double x){
   double xa    = TMath::Power(x,fAlpha_q); 
   double dminus = (1./xa)*( fCd*TMath::Power(1.-x,5.) + fCd_pr*TMath::Power(1.-x,5.)*TMath::Power(TMath::Log(1-x),2.) 
                           + fDd*TMath::Power(1.-x,6.) ); 
   return dminus; 
}
//______________________________________________________________________________
double Avakian::get_s_plus(double x){
   double xa    = TMath::Power(x,fAlpha_q); 
   double splus = (1./xa)*( fAs*TMath::Power(1.-x,5.) + fBs*TMath::Power(1.-x,6.) ); 
   return splus; 
}
//______________________________________________________________________________
double Avakian::get_s_minus(double x){
   double xa     = TMath::Power(x,fAlpha_q); 
   double sminus = (1./xa)*( fCs*TMath::Power(1.-x,7.) + fDs*TMath::Power(1.-x,8.) ); 
   return sminus; 
}
