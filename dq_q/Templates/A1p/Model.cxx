// Make a plot of the model A1n data 

#include <cstdlib> 
#include <iostream>
#include <vector>
#include <string> 

#include "./src/CSVManager.cpp"
#include "./src/BBS.cpp"
#include "./src/Avakian.cpp"

int Model(){

   char outpath[200]; 

   Avakian *model = new Avakian();
   sprintf(outpath,"avakian.dat"); 

   // BBS *model     = new BBS(); 
   // sprintf(outpath,"bbs_Q2_4.dat"); 

   const int NP = 100;
   double xMin = 1E-5;
   double xMax = 1;
   double step = (xMax-xMin)/( (double)NP );

   double arg_x=0,arg_y=0; 
   std::vector<double> xx,yy;  
   for(int i=0;i<NP;i++){
      arg_x = xMin + ( (double)i )*step;
      arg_y = model->get_A1n(arg_x);
      xx.push_back(arg_x);  
      yy.push_back(arg_y);  
   } 

   CSVManager *csv = new CSVManager("tsv");
   csv->InitTable(NP,2);
   csv->SetHeader("x,A1n"); 
   csv->SetColumn<double>(0,xx); 
   csv->SetColumn<double>(1,yy); 
   csv->WriteFile(outpath); 

   delete model;
   delete csv;

   return 0;
}
