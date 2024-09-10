// Make a plot of the E12-06-110 uncertainties
// along the Avakian curve 

#include <cstdlib> 
#include <iostream>
#include <vector>
#include <string> 

#include "./src/CSVManager.cpp"
#include "./src/Avakian.cpp"

int ModelExp(){

   CSVManager *data = new CSVManager("tsv");
   data->ReadFile("./data/E12-06-110_spect.dat",true); 

   std::vector<double> x,hms,shms; 
   data->GetColumn_byIndex<double>(0,x);  
   data->GetColumn_byIndex<double>(1,hms); 
   data->GetColumn_byIndex<double>(2,shms); 

   Avakian *model = new Avakian();

   double arg_x=0,arg_y=0,arg_ey=0,arg_ey_inv=0,T1=0,T2=0;
   std::vector<double> ey,y,Q2,syst;
   const int N = x.size(); 
   for(int i=0;i<N;i++){
      arg_y = model->get_A1n(x[i]);
      T1 = 0;
      T2 = 0;
      if(hms[i]!=0)  T1 = 1./(hms[i]*hms[i]);  
      if(shms[i]!=0) T2 = 1./(shms[i]*shms[i]);  
      arg_ey_inv = T1 + T2; 
      arg_ey = TMath::Sqrt(1./arg_ey_inv);  
      y.push_back(arg_y); 
      ey.push_back(arg_ey); 
      Q2.push_back(0); 
      syst.push_back(0); 
   } 

   CSVManager *csv = new CSVManager("tsv");
   csv->InitTable(N,5);
   csv->SetHeader("x,Q2,A1n,stat,syst"); 
   csv->SetColumn<double>(0,x); 
   csv->SetColumn<double>(1,Q2); 
   csv->SetColumn<double>(2,y); 
   csv->SetColumn<double>(3,ey); 
   csv->SetColumn<double>(4,syst); 
   csv->WriteFile("JLab_E12-06-110.dat"); 

   delete model;
   delete data;
   delete csv;

   return 0;
}
