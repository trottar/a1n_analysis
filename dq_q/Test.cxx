// test kinematics namespace and R1998 class 

#include <cstdlib>
#include <iostream> 

#include "./src/R1998.cpp"
#include "./src/Kinematics.cpp"
#include "./src/Graph.cpp"

int Test(){

   R1998 *R = new R1998(); 

   const int N = 20; 
   double Es    = 4.74; // in GeV 
   double EpMin = 0.6;  // in GeV 
   double EpMax = 1.7;  // in GeV  
   double epStep = (EpMax-EpMin)/( (double)N );
 
   double Q2 = 4.0;
   double th = 45.0; // deg    
   double ix=0,r=0,iD=0,iep=0;
   std::vector<double> XX,RR,DD; 
   for(int i=0;i<N;i++){
      iep = EpMin + ( (double)i )*epStep;
      ix  = Kinematics::GetXbj(Es,iep,th); 
      r   = R->GetR(ix,Q2);
      iD  = Kinematics::GetD(Es,iep,th,r); 
      XX.push_back(ix);
      RR.push_back(r);  
      DD.push_back(iD);  
   }
 
   TGraph *g = graph_df::GetTGraph(XX,DD); 
   graph_df::SetParameters(g,20,kBlack);

   TString Title = Form("D(x,Q^{2} = %.1lf GeV^{2})",Q2); 

   TCanvas *c1 = new TCanvas("c1","Kinematics Test",1000,800);

   c1->cd(1);
   g->Draw("alp");
   graph_df::SetLabels(g,Title,"x","D");  
   g->Draw("alp");
   c1->Update();

   return 0;
}
