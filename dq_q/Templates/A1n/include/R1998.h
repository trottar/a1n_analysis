#ifndef R1998_H
#define R1998_H 

// R = sigma_L/sigma_T fit as a function of x and Q2 
// Reference: Phys. Lett. B 452, 194 (1999) 

class R1998 {
   private:
      // fit parameters 
      double *fa,*fb,*fc;  
      // functions 
      double GetTheta(double x,double Q2); 
      double GetRa(double x,double Q2); 
      double GetRb(double x,double Q2); 
      double GetRc(double x,double Q2); 

   public:
      R1998();
      ~R1998();

      double GetR(double x,double Q2); 
}; 

#endif 
