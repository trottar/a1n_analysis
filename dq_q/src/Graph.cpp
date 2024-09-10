#include "../include/Graph.h"

namespace graph_df {
   //______________________________________________________________________________
   TGraph *GetTGraph(std::vector<double> x,std::vector<double> y){
      const int N = x.size();
      TGraph *g = new TGraph(N,&x[0],&y[0]);
      return g;
   }
   //______________________________________________________________________________
   TGraphErrors *GetTGraphErrors(std::vector<double> x,std::vector<double> y,std::vector<double> ey){
      const int N = x.size();
      std::vector<double> ex;
      for(int i=0;i<N;i++) ex.push_back(0);
      TGraphErrors *g = new TGraphErrors(N,&x[0],&y[0],&ex[0],&ey[0]);
      return g;
   }
   //______________________________________________________________________________
   TGraphErrors *GetTGraphErrors(std::vector<double> x,std::vector<double> ex,std::vector<double> y,std::vector<double> ey){
      const int N = x.size();
      TGraphErrors *g = new TGraphErrors(N,&x[0],&y[0],&ex[0],&ey[0]);
      return g;
   }
   //______________________________________________________________________________
   TGraphMultiErrors *GetTGraphMultiErrors(std::vector<double> x,std::vector<double> ex,std::vector<double> y,std::vector<double> staty,std::vector<double> pdfy,std::vector<double> systy){
      const int N = x.size();
      TGraphMultiErrors *g = new TGraphMultiErrors("g","projections",N,&x[0],&y[0],&ex[0],&ex[0],&staty[0],&staty[0]);
      g->AddYError(N, &pdfy[0], &pdfy[0]);
      g->AddYError(N, &systy[0], &systy[0]);
      return g;
   }
   //______________________________________________________________________________
   TGraphAsymmErrors *GetTGraphAsymmErrors(std::vector<double> x,std::vector<double> y,std::vector<double> eyl,std::vector<double> eyh){
      const int N = x.size();
      std::vector<double> exl,exh;
      for(int i=0;i<N;i++){
	 exl.push_back(0);
	 exh.push_back(0);
      }
      TGraphAsymmErrors *g = new TGraphAsymmErrors(N,&x[0],&y[0],&exl[0],&exh[0],&eyl[0],&eyh[0]);
      return g;
   }
   //______________________________________________________________________________
   void SetLabelSizes(TGraph *g,double xSize,double ySize,double offset){
      g->GetXaxis()->SetTitleSize(xSize);
      g->GetXaxis()->SetLabelSize(xSize);
      g->GetYaxis()->SetTitleSize(ySize);
      g->GetYaxis()->SetLabelSize(ySize);
      g->GetYaxis()->SetTitleOffset(offset);
   }
   //______________________________________________________________________________
   void SetLabelSizes(TGraphErrors *g,double xSize,double ySize,double offset){
      g->GetXaxis()->SetTitleSize(xSize);
      g->GetXaxis()->SetLabelSize(xSize);
      g->GetYaxis()->SetTitleSize(ySize);
      g->GetYaxis()->SetLabelSize(ySize);
      g->GetYaxis()->SetTitleOffset(offset);
   } 
   //______________________________________________________________________________
   void SetLabelSizes(TMultiGraph *g,double xSize,double ySize,double offset){
      g->GetXaxis()->SetTitleSize(xSize);
      g->GetXaxis()->SetLabelSize(xSize);
      g->GetYaxis()->SetTitleSize(ySize);
      g->GetYaxis()->SetLabelSize(ySize);
      g->GetYaxis()->SetTitleOffset(offset);
   }
   //______________________________________________________________________________
   void SetLabels(TGraphErrors *g,TString Title,TString xAxisTitle,TString yAxisTitle){
      g->SetTitle(Title);
      g->GetXaxis()->SetTitle(xAxisTitle);
      g->GetXaxis()->CenterTitle();
      g->GetYaxis()->SetTitle(yAxisTitle);
      g->GetYaxis()->CenterTitle();
   }
   //______________________________________________________________________________
   void SetLabels(TGraph *g,TString Title,TString xAxisTitle,TString yAxisTitle){
      g->SetTitle(Title);
      g->GetXaxis()->SetTitle(xAxisTitle);
      g->GetXaxis()->CenterTitle();
      g->GetYaxis()->SetTitle(yAxisTitle);
      g->GetYaxis()->CenterTitle();
   }
   //______________________________________________________________________________
   void SetLabels(TMultiGraph *g,TString Title,TString xAxisTitle,TString yAxisTitle){
      g->SetTitle(Title);
      g->GetXaxis()->SetTitle(xAxisTitle);
      g->GetXaxis()->CenterTitle();
      g->GetYaxis()->SetTitle(yAxisTitle);
      g->GetYaxis()->CenterTitle();
   }
   //______________________________________________________________________________
   void SetParameters(TGraphErrors *g,int mStyle,int color,double size,int width){
      g->SetMarkerStyle(mStyle);
      g->SetMarkerColor(color);
      g->SetLineColor(color);
      g->SetMarkerSize(size);
      g->SetLineWidth(width);
   }
   //______________________________________________________________________________
   void SetParameters(TGraph *g,int mStyle,int color,double size,int width){
      g->SetMarkerStyle(mStyle);
      g->SetMarkerColor(color);
      g->SetLineColor(color);
      g->SetMarkerSize(size);
      g->SetLineWidth(width);
   }
   //______________________________________________________________________________
   void SetParameters(TGraph *g,int mLineStyle,int mFillStyle,int color,int width){
      g->SetLineColor(color);
      g->SetLineStyle(mLineStyle);
      g->SetFillColor(color);
      g->SetFillStyle(mFillStyle);
      g->SetLineWidth(width);
   }
   //______________________________________________________________________________
   void SetParameters(TGraphErrors *g,int mLineStyle,int mFillStyle,int color,int width){
      g->SetLineColor(color);
      g->SetLineStyle(mLineStyle);
      g->SetFillColor(color);
      g->SetFillStyle(mFillStyle);
      g->SetLineWidth(width);
    }
   //______________________________________________________________________________
   void SetParameters(TGraph *g,int mLineStyle,int color,int width){
      g->SetLineColor(color);
      g->SetLineStyle(mLineStyle);
      g->SetLineWidth(width);
   }
   //______________________________________________________________________________
   void SetParameters(TGraphErrors *g,int mLineStyle,int color,int width){
      g->SetLineColor(color);
      g->SetLineStyle(mLineStyle);
      g->SetLineWidth(width);
   }
}
