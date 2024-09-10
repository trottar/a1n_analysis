#include "../include/R1998.h"
//______________________________________________________________________________
R1998::R1998(){
   const int NP = 6;
   fa = new double[NP]; 
   fb = new double[NP]; 
   fc = new double[NP]; 
   // a coefficients 
   fa[0] = 0.0485;  fa[1] = 0.5470; fa[2] = 2.0621; 
   fa[3] = -0.3804; fa[4] = 0.5090; fa[5] = -0.0285; 
   // b coefficients 
   fb[0] = 0.0481;  fb[1] = 0.6114; fb[2] = -0.3509;
   fb[3] = -0.4611; fb[4] = 0.7172; fb[5] = -0.0317;
   // c coefficients
   fc[0] = 0.0577;  fc[1] = 0.4644;   fc[2] = 1.8288;
   fc[3] = 12.3708; fc[4] = -43.1043; fc[5] = 41.7415;
}
//______________________________________________________________________________
R1998::~R1998(){
   delete fa; 
   delete fb; 
   delete fc; 
}
//________________________________________________________________________
double R1998::GetR(double x, double Q2){
   double Ra = GetRa(x,Q2);  
   double Rb = GetRb(x,Q2);  
   double Rc = GetRc(x,Q2); 
   double R  = (1./3.)*(Ra+Rb+Rc);  
   return R;
}
//________________________________________________________________________
double R1998::GetRa(double x,double Q2){
   double Q8   = Q2*Q2*Q2; 
   double TH   = GetTheta(x,Q2); 
   double T1   = fa[0]/(log(Q2/0.04))*TH;
   double T2_a = fa[1]/(pow(Q8 + pow(fa[2],4),1./4.));  
   double T2_b = (1 + fa[3]*x+ fa[4]*x*x)*pow(x,fa[5]);  
   double T2   = T2_a*T2_b;
   double Ra   = T1 + T2;
   return Ra; 
}
//________________________________________________________________________
double R1998::GetRb(double x,double Q2){
   double Q4   = Q2*Q2; 
   double TH   = GetTheta(x,Q2); 
   double T1   = fb[0]/(log(Q2/0.04))*TH;
   double T2_a = fb[1]/Q2+ fb[2]/(Q4 + pow(0.3,2.));  
   double T2_b = (1 + fb[3]*x+ fb[4]*x*x)*pow(x,fb[5]);  
   double T2   = T2_a*T2_b;
   double Rb   = T1 + T2;
   return Rb; 
}
//________________________________________________________________________
double R1998::GetRc(double x,double Q2){
   double Q4   = Q2*Q2;
   double Q2_thr = fc[3]*x + fc[4]*x*x + fc[5]*x*x*x;  
   double TH   = GetTheta(x,Q2); 
   double T1   = fc[0]/(log(Q2/0.04))*TH;
   double T2_a = fc[1]; 
   double arg  = pow(Q2-Q2_thr,2.) + pow(fc[2],2.); 
   double T2_b = pow(arg,-0.5);  
   double T2   = T2_a*T2_b;
   double Rc   = T1 + T2;
   return Rc; 
}
//________________________________________________________________________
double R1998::GetTheta(double x,double Q2){
   // construct TH(x,Q2) 
   double TH_t1  = 1.;
   double TH_t2a = 12.*Q2/(Q2+1.);   
   double TH_t2b_num = pow(0.125,2.);   
   double TH_t2b_den = pow(0.125,2.) + pow(x,2.);  
   double TH_t2b = TH_t2b_num/TH_t2b_den; 
   double TH     = TH_t1 + TH_t2a*TH_t2b; 
   return TH;
}
