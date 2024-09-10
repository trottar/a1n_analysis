// compute A1 given Apara and Aperp 

#include <cstdlib> 
#include <iostream>

#include "./src/Kinematics.cpp"
#include "./src/R1998.cpp"

double GetA1(double Es,double Ep,double th,double Apara,double Aperp); 

int A1(){

   return 0;
}
//______________________________________________________________________________
double GetA1(double Es,double Ep,double th,double Apara,double Aperp){
   // get kinematic values
   R1998 *R98 = new R1998();
   double x   = Kinematics::GetXbj(Es,Ep,th); 
   double Q2  = Kinematics::GetQ2(Es,Ep,th); 
   double r   = R98->GetR(x,Q2);
   double D   = Kinematics::GetD(Es,Ep,th,r); 
   double d   = Kinematics::Getd(Es,Ep,th); 
   double eta = Kinematics::GetEta(Es,Ep,th); 
   double xi  = Kinematics::GetXi(Es,Ep,th);
   delete R98;  
   // construct A1  
   double T1_num  = 1.;  
   double T1_den  = D*(1 + eta*xi); 
   double T1=0;
   if(T1_den!=0){
      T1 = T1_num/T1_den;
   }else{
      std::cout << "[A1::GetA1]: ERROR!  Denominator for first term is zero!" << std::endl; 
   }
   double T2_num = (-1.)*eta; 
   double T2_den = d*(1 + eta*xi);
   double T2=0;
   if(T2_den!=0){
      T2 = T2_num/T2_den;
   }else{
      std::cout << "[A1::GetA1]: ERROR!  Denominator for second term is zero!" << std::endl; 
   } 
   double a1 = T1*Apara + T2*Aperp;
   return a1;  
}
