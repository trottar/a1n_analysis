// Make a plot of the world A1p data 

#include <cstdlib> 
#include <iostream>
#include <vector>
#include <string> 

#include "TLine.h"
#include "TStyle.h"

#include "./src/CSVManager.cpp"
#include "./src/Graph.cpp"
#include "./src/BBS.cpp"

bool gIsDebug = true;

TGraph *GetModelTGraph(CSVManager *model); 
TGraphErrors *GetTGraphErrors(std::string exp,CSVManager *data); 

int Plot(){

   // for plotting 
   bool hasHeader = false;
   CSVManager *data = new CSVManager("tsv");

   TMultiGraph *mg = new TMultiGraph();
   TLegend *L      = new TLegend(0.6,0.6,0.8,0.8);
   TLegend *LM     = new TLegend(0.6,0.6,0.8,0.8);
 
   L->SetBorderSize(0); 
   LM->SetBorderSize(0);  

   TString label;
   char inpath[200]; 

   // load models
   CSVManager *par_model = new CSVManager("csv");
   par_model->ReadFile("./input/model-list.csv",true);
   
   std::vector<std::string> mLabel,mPath;
   par_model->GetColumn_byName_str("label",mLabel);  
   par_model->GetColumn_byName_str("path" ,mPath );  
  
   std::vector<int> mLineStyle,mColor; 
   par_model->GetColumn_byName<int>("line_style",mLineStyle);  
   par_model->GetColumn_byName<int>("color"     ,mColor    );  

   CSVManager *model= new CSVManager("tsv"); 

   const int NM = mLabel.size();
   TGraph **gm = new TGraph*[NM];

   double mSize = 1.2; 
   int mMarker=20;

   for(int i=0;i<NM;i++){
      // load model
      sprintf(inpath,"./models/%s",mPath[i].c_str()); 
      model->ReadFile(inpath,true); 
      // create plot
      if(mLabel[i].compare("DSE(realistic)")==0){
	 mMarker = 42;
         mSize   = 2;
      }else if(mLabel[i].compare("DSE(contact)")==0){
	 mMarker = 28;
         mSize   = 2;
      }else{
	 mMarker = 20;
         mSize   = 1;
      } 
      gm[i] = GetModelTGraph(model);
      graph_df::SetParameters(gm[i],mMarker,mColor[i],mSize,3); 
      gm[i]->SetLineStyle(mLineStyle[i]);      
      // add to multigraph object
      label = Form("%s",mLabel[i].c_str()); 
      if(mLabel[i].compare("CQM")==0){
	 gm[i]->SetFillColorAlpha(mColor[i],0.50); // make fill color transparent 
	 mg->Add(gm[i],"f");                       // draw as a filled object 
	 LM->AddEntry(gm[i],label,"l"); 
      }else if(mLabel[i].compare("DSE(realistic)")==0){
	 mg->Add(gm[i],"p");
	 LM->AddEntry(gm[i],label,"p"); 
      }else if(mLabel[i].compare("DSE(contact)")==0){
	 mg->Add(gm[i],"p");
	 LM->AddEntry(gm[i],label,"p"); 
      }else{ 
	 mg->Add(gm[i],"l");
	 LM->AddEntry(gm[i],label,"l"); 
      } 
      // set up for next model 
      model->ClearData(); 
   }

   // read in experiment names
   char inpath_par[200];
   sprintf(inpath_par,"./input/exp-list.csv"); 
   CSVManager *par = new CSVManager("csv"); 
   par->ReadFile(inpath_par,true);

   std::vector<std::string> exp; 
   par->GetColumn_byName_str("exp",exp);

   std::vector<int> marker,color; 
   par->GetColumn_byName<int>("marker",marker);  
   par->GetColumn_byName<int>("color" ,color);  

   const int N = exp.size(); 
   TGraphErrors **g = new TGraphErrors*[N]; 

   mSize = 1.2;
 
   // loop over experiments and add to plots 
   for(int i=0;i<N;i++){
      // load data from a given experiment 
      sprintf(inpath,"./data/%s.dat",exp[i].c_str());
      data->ReadFile(inpath,hasHeader);
      // create the graph
      g[i] = GetTGraphErrors(exp[i],data);
      graph_df::SetParameters(g[i],marker[i],color[i],mSize);
      // add to multigraph object 
      mg->Add(g[i],"p");
      // create a label for the legend  
      label = Form("%s",exp[i].c_str()); 
      if(exp[i].compare("JLab_E12-06-110_hi_Espec")==0) label = Form("JLab_E12-06-110 (lo E'; no W)");  
      if(exp[i].compare("JLab_E12-06-110_lo_Espec")==0)        label = Form("JLab E12-06-110 (hi E'; no W)");
      if(exp[i].compare("JLab_E12-06-110_no-W")!=0) L->AddEntry(g[i],label,"p");
      // set up for next data set 
      data->ClearData(); 
   } 

   // make the plot 
   TString xAxisTitle = Form("x"); 
   TString yAxisTitle = Form("A_{1}^{p}");

   double xMin = 0;
   double xMax = 1; 
   double yMin = -0.2;
   double yMax = 1.2;
 
   TLine *xAxisLine = new TLine(xMin,0,xMax,0); 

   TCanvas *c1 = new TCanvas("c1","A1p Data",1000,800); 
   c1->cd();
   
   mg->Draw("a");
   graph_df::SetLabels(mg,"",xAxisTitle,yAxisTitle);
   mg->GetXaxis()->SetLimits(xMin,xMax); 
   mg->GetYaxis()->SetRangeUser(yMin,yMax); 
   mg->Draw("a");
   xAxisLine->Draw("same"); 
   L->Draw("same");
   LM->Draw("same");
   c1->Update();

   // delete CSV objects
   delete par;
   delete data;
   delete par_model;
   delete model;

   return 0;
}
//______________________________________________________________________________
TGraph *GetModelTGraph(CSVManager *model){
   // get a TGraph object from the model data
   std::vector<double> x,y;
   model->GetColumn_byIndex<double>(0,x); 
   model->GetColumn_byIndex<double>(1,y);

   TGraph *g = graph_df::GetTGraph(x,y);
   return g; 
}
//______________________________________________________________________________
TGraphErrors *GetTGraphErrors(std::string exp,CSVManager *data){
   // make a TGraphErrors object for experiment data  
   std::vector<double> x,xmax,ex,y,stat,syst,syst_lo,ey; 
   data->GetColumn_byIndex<double>(0,x); 
   data->GetColumn_byIndex<double>(2,y); 
   data->GetColumn_byIndex<double>(3,stat); 
   data->GetColumn_byIndex<double>(4,syst);
   
   const int N = x.size();
   for(int i=0;i<N;i++) ex.push_back(0); 
   
   if(gIsDebug) std::cout << exp << std::endl;

   if(exp.compare("JLab_E99117")==0){
      // need to handle the asymmetric errors
      // will take a conservative approach and use the larger uncertainty 
      data->GetColumn_byIndex<double>(5,syst_lo); 
      for(int i=0;i<N;i++){
	 if( fabs(syst_lo[i])>fabs(syst[i]) ) syst[i] = fabs(syst_lo[i]); 
      }
   }
   // else if(exp.compare("JLab_E22")==0){
   //    // projection at E = 22 GeV
   //    // columns are xmin, xmax, A1n, stat, syst 
   //    data->GetColumn_byIndex<double>(1,xmax);
   //    // bin width
   //    // for(int i=0;i<N;i++) ex[i] = 0.5*(xmax[i]-x[i]); 
   //    // x central value 
   //    for(int i=0;i<N;i++) x[i]  = 0.5*(x[i] + xmax[i]);
   // }

   double arg=0;
   for(int i=0;i<N;i++){
      arg = TMath::Sqrt( stat[i]*stat[i] + syst[i]*syst[i] );
      ey.push_back(arg); 
      if(gIsDebug) std::cout << Form("x = %.3lf, A1p = %.3lf ± %.3lf ± %.3lf (%.3lf)",x[i],y[i],stat[i],syst[i],ey[i]) << std::endl;
   }
   if(gIsDebug) std::cout << "------------------" << std::endl;

   TGraphErrors *g = graph_df::GetTGraphErrors(x,ex,y,ey);
   return g; 
}
