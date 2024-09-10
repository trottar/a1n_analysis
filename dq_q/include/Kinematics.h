#ifndef KINEMATICS_HH 
#define KINEMATICS_HH 

// kinematics namespace for common variables like x, Q2, W, etc

#include <cstdlib>
#include <iostream>
#include <cmath>

#include "constants.h"

namespace Kinematics {
   double GetQ2(double Es,double Ep,double th);
   double GetEpsilon(double Es,double Ep,double th);
   double GetEp_Elastic(double Es,double th,double M=constants::proton_mass);
   double GetW(double Es,double Ep,double th,double M=constants::proton_mass);
   double GetXbj(double Es,double Ep,double th,double M=constants::proton_mass);

   double GetD(double Es,double Ep,double th,double R=1); // NOTE: user needs to supply R(x,Q2)! 
   double Getd(double Es,double Ep,double th,double R=1); // NOTE: user needs to supply R(x,Q2)!
   double GetEta(double Es,double Ep,double th); 
   double GetXi(double Es,double Ep,double th); 

}

#endif
