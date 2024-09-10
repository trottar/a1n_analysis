#include "../include/BBS.h"
//______________________________________________________________________________
BBS::BBS(){
   // quark charge 
   fQu    = 2./3.; 
   fQd    = -1./3.; 
   fQs    = -1./3.;
   // parameters  
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
}
//______________________________________________________________________________
BBS::~BBS(){

}
//______________________________________________________________________________
double BBS::get_A1p(double x){
   double g1p = get_g1p(x);
   double F1p = get_F1p(x);
   double A1p = g1p/F1p;
   return A1p;
}
//______________________________________________________________________________
double BBS::get_A1n(double x){
   double g1n = get_g1n(x);
   double F1n = get_F1n(x);
   double A1n = g1n/F1n;
   return A1n;
}
//______________________________________________________________________________
double BBS::get_g1p(double x){
   double g1p = 0.5*( fQu*fQu*get_delta_u_uBar(x) + fQd*fQd*get_delta_d_dBar(x) + fQs*fQs*get_delta_s_sBar(x) ); 
   return g1p;
}
//______________________________________________________________________________
double BBS::get_F1p(double x){
   double F1p = 0.5*( fQu*fQu*get_u_uBar(x) + fQd*fQd*get_d_dBar(x) + fQs*fQs*get_s_sBar(x) ); 
   return F1p;
}
//______________________________________________________________________________
double BBS::get_g1n(double x){
   // NOTE the charges!
   double g1n = 0.5*( fQd*fQd*get_delta_u_uBar(x) + fQu*fQu*get_delta_d_dBar(x) + fQs*fQs*get_delta_s_sBar(x) ); 
   return g1n;
}
//______________________________________________________________________________
double BBS::get_F1n(double x){
   // NOTE the charges!
   double F1n = 0.5*( fQd*fQd*get_u_uBar(x) + fQu*fQu*get_d_dBar(x) + fQs*fQs*get_s_sBar(x) ); 
   return F1n;
}
//______________________________________________________________________________
double BBS::get_delta_u_uBar(double x){
   double xa = TMath::Power(x,fAlpha_q); 
   double f  = (1./xa)*TMath::Power(1.-x,3.)*( fAu + fBu*(1.-x) - fCu*TMath::Power(1.-x,2.) - fDu*TMath::Power(1.-x,3.) );  
   return f;  
}
//______________________________________________________________________________
double BBS::get_delta_d_dBar(double x){
   double xa = TMath::Power(x,fAlpha_q); 
   double f  = (1./xa)*TMath::Power(1.-x,3.)*( fAd + fBd*(1.-x) - fCd*TMath::Power(1.-x,2.) - fDd*TMath::Power(1.-x,3.) );  
   return f;  
}
//______________________________________________________________________________
double BBS::get_delta_s_sBar(double x){
   double xa = TMath::Power(x,fAlpha_q); 
   double f  = (1./xa)*TMath::Power(1.-x,5.)*( fAs + fBs*(1.-x) - fCs*TMath::Power(1.-x,2.) - fDs*TMath::Power(1.-x,3.) );  
   return f;  
}
//______________________________________________________________________________
double BBS::get_u_uBar(double x){
   double xa = TMath::Power(x,fAlpha_q); 
   double f  = (1./xa)*TMath::Power(1.-x,3.)*( fAu + fBu*(1.-x) + fCu*TMath::Power(1.-x,2.) + fDu*TMath::Power(1.-x,3.) );  
   return f;  
}
//______________________________________________________________________________
double BBS::get_d_dBar(double x){
   double xa = TMath::Power(x,fAlpha_q); 
   double f  = (1./xa)*TMath::Power(1.-x,3.)*( fAd + fBd*(1.-x) + fCd*TMath::Power(1.-x,2.) + fDd*TMath::Power(1.-x,3.) );  
   return f;  
}
//______________________________________________________________________________
double BBS::get_s_sBar(double x){
   double xa = TMath::Power(x,fAlpha_q); 
   double f  = (1./xa)*TMath::Power(1.-x,5.)*( fAs + fBs*(1.-x) + fCs*TMath::Power(1.-x,2.) + fDs*TMath::Power(1.-x,3.) );  
   return f;  
}
//______________________________________________________________________________
double BBS::get_g(double x){
   double xa = TMath::Power(x,fAlpha_g);
   double f  = (1./xa)*TMath::Power(1.-x,4.)*( fAg + fBg*(1.-x) )*( 1. + TMath::Power(1.-x,2.) );
   return f;  
}
//______________________________________________________________________________
double BBS::get_delta_g(double x){
   double xa = TMath::Power(x,fAlpha_g);
   double f  = (1./xa)*TMath::Power(1.-x,4.)*( fAg + fBg*(1.-x) )*( 1. - TMath::Power(1.-x,2.) );
   return f;  
}
