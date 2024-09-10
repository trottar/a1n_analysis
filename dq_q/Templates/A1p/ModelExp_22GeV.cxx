// Make a plot of the E12-06-122 uncertainties
// along the Avakian curve 

#include <cstdlib> 
#include <iostream>
#include <vector>
#include <string> 

#include "./src/CSVManager.cpp"
#include "./src/Avakian.cpp"

int ModelExp_22GeV(){

   CSVManager *data = new CSVManager("tsv");
   data->ReadFile("./data/JLab_E22_07-13-21.dat",true); 

   std::vector<double> xmin,xmax,stat,syst; 
   data->GetColumn_byIndex<double>(0,xmin);  
   data->GetColumn_byIndex<double>(1,xmax); 
   data->GetColumn_byIndex<double>(3,stat); 
   data->GetColumn_byIndex<double>(4,syst); 

   Avakian *model = new Avakian();

   double arg_x=0,arg_y=0;
   std::vector<double> x,y,Q2;
   const int N = xmin.size(); 
   for(int i=0;i<N;i++){
      arg_x = 0.5*(xmax[i] + xmin[i]);
      arg_y = model->get_A1n(arg_x);
      // T1 = 0;
      // T2 = 0;
      // if(hms[i]!=0)  T1 = 1./(hms[i]*hms[i]);  
      // if(shms[i]!=0) T2 = 1./(shms[i]*shms[i]);  
      // arg_ey_inv = T1 + T2; 
      // arg_ey = TMath::Sqrt(1./arg_ey_inv);  
      x.push_back(arg_x); 
      y.push_back(arg_y); 
      Q2.push_back(0); 
   } 

   CSVManager *csv = new CSVManager("tsv");
   csv->InitTable(N,5);
   csv->SetHeader("x,Q2,A1n,stat,syst"); 
   csv->SetColumn<double>(0,x); 
   csv->SetColumn<double>(1,Q2); 
   csv->SetColumn<double>(2,y); 
   csv->SetColumn<double>(3,stat); 
   csv->SetColumn<double>(4,syst); 
   csv->WriteFile("JLab_E22_along-pqcd-oam_07-13-21.dat"); 

   delete model;
   delete data;
   delete csv;

   return 0;
}
