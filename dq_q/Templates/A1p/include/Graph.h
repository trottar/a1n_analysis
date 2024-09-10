#ifndef GRAPH_DF_H
#define GRAPH_DF_H

#include "TCanvas.h"
#include "TLegend.h"
#include "TString.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"

namespace graph_df {
   TGraph *GetTGraph(std::vector<double>,std::vector<double>);
   TGraphErrors *GetTGraphErrors(std::vector<double> x,std::vector<double> y,std::vector<double> ey);
   TGraphErrors *GetTGraphErrors(std::vector<double> x,std::vector<double> ex,std::vector<double> y,std::vector<double> ey);
   TGraphAsymmErrors *GetTGraphAsymmErrors(std::vector<double> x,std::vector<double> y,std::vector<double> eyl,std::vector<double> eyh);
   void SetLabelSizes(TGraph *g,double xSize,double ySize,double offset=0.5); 
   void SetLabelSizes(TGraphErrors *g,double xSize,double ySize,double offset=0.5); 
   void SetLabelSizes(TMultiGraph *g,double xSize,double ySize,double offset=0.5); 
   void SetParameters(TGraphErrors *g,int mStyle,int color,double size=1.0,int width=1);
   void SetParameters(TGraph *g,int mStyle,int color,double size=1.0,int width=1);
   void SetLabels(TGraph *g,TString Title,TString xAxisTitle,TString yAxisTitle);
   void SetLabels(TGraphErrors *g,TString Title,TString xAxisTitle,TString yAxisTitle);
   void SetLabels(TMultiGraph *g,TString Title,TString xAxisTitle,TString yAxisTitle);
}

#endif 
