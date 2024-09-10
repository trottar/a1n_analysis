// Make a plot of the world dq/q data

#include <cstdlib> 
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "TLine.h"
#include "TStyle.h"

#include "./src/CSVManager.cpp"
#include "./src/Graph.cpp"
#include "./src/BBS.cpp"

bool gIsDebug = true;
TGraph *GetModelTGraph(CSVManager *model);
TGraph *GetUpModelTGraphLine(CSVManager *model);
TGraph *GetDownModelTGraphLine(CSVManager *model);
TGraph *GetUpModelTGraphFill(CSVManager *model);
TGraph *GetDownModelTGraphFill(CSVManager *model);
TGraph *GetUpPDFError(CSVManager *data);
TGraph *GetDownPDFError(CSVManager *data);
TGraphErrors *GetTGraphErrors(std::string exp,CSVManager *data);
TGraphErrors *GetUpTGraphErrors(std::string exp,CSVManager *data,bool ifProj);
TGraphErrors *GetDownTGraphErrors(std::string exp,CSVManager *data,bool ifProj);
double downBBSOAM_func(double x);
double upBBSOAM_func(double x);
void PlotA1p();
void PlotA1n();
void PlotDQoQ();

int Plot() {
    PlotA1p();
    PlotA1n();
    PlotDQoQ();
    
    return 0;
}
//______________________________________________________________________________
void PlotA1p(){
    
    // for plotting
    bool hasHeader = false;
    CSVManager *model= new CSVManager("tsv");
    CSVManager *data = new CSVManager("tsv");
    CSVManager *proj = new CSVManager("tsv");
    
    CSVManager *par_model = new CSVManager("csv");
    CSVManager *par_data = new CSVManager("csv");
    CSVManager *par_proj = new CSVManager("csv");
    
    TString label;
    char inpath_model[200];
    char inpath_data[200];
    char inpath_proj[200];

    TMultiGraph *mg = new TMultiGraph();
    TLegend *L      = new TLegend(0.6,0.6,0.8,0.8); // experiment and projection legend
    TLegend *LM     = new TLegend(0.6,0.6,0.8,0.8); // model legend
  
    L->SetBorderSize(0);
    LM->SetBorderSize(0);

    // load models
    par_model->ReadFile("./input/input_model/model-list-A1p_noclas.csv",true);
    
    std::vector<std::string> mLabel,mPath;
    par_model->GetColumn_byName_str("label",mLabel);
    par_model->GetColumn_byName_str("path" ,mPath );
   
    std::vector<int> mLineStyle,mColor;
    par_model->GetColumn_byName<int>("line_style",mLineStyle);
    par_model->GetColumn_byName<int>("color"     ,mColor    );

    const int NM = mLabel.size();
    TGraph **gm = new TGraph*[NM];

    double mSize = 1.2;
    int mMarker=20;

    for(int i=0;i<NM;i++){
       // load model
       sprintf(inpath_model,"./models/models_A1p/%s",mPath[i].c_str());
       model->ReadFile(inpath_model,true);
       // create plot
       if(mLabel[i].compare("DSE(realistic)")==0){
          mMarker = 46;
          mSize   = 2;
       }else if(mLabel[i].compare("DSE(contact)")==0){
          mMarker = 28;
          mSize   = 2;
       }else if(mLabel[i].compare("Cheng et al.")==0){
          mMarker = 42;
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
       }else if(mLabel[i].compare("Cheng et al.")==0){
          mg->Add(gm[i],"p");
          LM->AddEntry(gm[i],label,"p");
       }else{
          mg->Add(gm[i],"l");
          LM->AddEntry(gm[i],label,"l");
       }
       // set up for next model
       model->ClearData();
    }

    // load experiments
    par_data->ReadFile("./input/input_exp/exp-list-A1p.csv",true);

    std::vector<std::string> exp;
    par_data->GetColumn_byName_str("exp",exp);

    std::vector<int> marker,color;
    par_data->GetColumn_byName<int>("marker",marker);
    par_data->GetColumn_byName<int>("color" ,color);

    const int N = exp.size();
    TGraphErrors **g = new TGraphErrors*[N];

    mSize = 1.2;
  
    // loop over experiments and add to plots
    for(int i=0;i<N;i++){
       // load data from a given experiment
       sprintf(inpath_data,"./data/data_A1p/%s.dat",exp[i].c_str());
       data->ReadFile(inpath_data,hasHeader);
       // create the graph
       g[i] = GetTGraphErrors(exp[i],data);
       graph_df::SetParameters(g[i],marker[i],color[i],mSize);
       // add to multigraph object
       mg->Add(g[i],"p");
       // create a label for the legend
       label = Form("%s",exp[i].c_str());
       if(exp[i].compare("JLab_E12-06-110_hi_Espec")==0) label = Form("JLab_E12-06-110 (lo E'; no W)");
       if(exp[i].compare("JLab_E12-06-110_lo_Espec")==0) label = Form("JLab E12-06-110 (hi E'; no W)");
       if(exp[i].compare("JLab_E12-06-110_no-W")!=0) L->AddEntry(g[i],label,"p");
       // set up for next data set
       data->ClearData();
    }
    
    // read in projections
    par_proj->ReadFile("./input/input_proj/proj_list/proj-list-A1p.csv",true);

    std::vector<std::string> projection;
    par_proj->GetColumn_byName_str("projection",projection);

    std::vector<int> projLine,projColor;
    par_proj->GetColumn_byName<int>("line_style",projLine);
    par_proj->GetColumn_byName<int>("color" ,projColor);

    const int NP = projection.size();
    TGraphErrors **pg = new TGraphErrors*[N];
  
    // loop over projections and add to plots
    for(int i=0;i<NP;i++){
       // load data from a given projection
       sprintf(inpath_proj,"./projections/projections_A1p/%s.dat",projection[i].c_str());
       std::cout << projection[i];
       proj->ReadFile(inpath_proj,true);
       // create the graph
       pg[i] = GetTGraphErrors(projection[i],proj);
       graph_df::SetParameters(pg[i],projLine[i],projColor[i],mSize);
       // add to multigraph object
       mg->Add(pg[i],"p");
       // create a label for the legend
       label = Form("%s (Proj.)",projection[i].c_str());
       L->AddEntry(pg[i],label,"p");
       // set up for next data set
       proj->ClearData();
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
    delete model;
    delete data;
    delete proj;
    
    delete par_model;
    delete par_data;
    delete par_proj;
}
//______________________________________________________________________________
void PlotA1n(){
    // for plotting
    bool hasHeader = false;
    CSVManager *model= new CSVManager("tsv");
    CSVManager *data = new CSVManager("tsv");
    CSVManager *proj = new CSVManager("tsv");
    
    CSVManager *par_model = new CSVManager("csv");
    CSVManager *par_data = new CSVManager("csv");
    CSVManager *par_proj = new CSVManager("csv");
    
    TString label;
    char inpath_model[200];
    char inpath_data[200];
    char inpath_proj[200];

    TMultiGraph *mg = new TMultiGraph();
    TLegend *L      = new TLegend(0.6,0.6,0.8,0.8);
    TLegend *LM     = new TLegend(0.6,0.6,0.8,0.8);
  
    L->SetBorderSize(0);
    LM->SetBorderSize(0);


    // load models
    par_model->ReadFile("./input/input_model/model-list-A1n_noclas.csv",true);
    
    std::vector<std::string> mLabel,mPath;
    par_model->GetColumn_byName_str("label",mLabel);
    par_model->GetColumn_byName_str("path" ,mPath );
   
    std::vector<int> mLineStyle,mColor;
    par_model->GetColumn_byName<int>("line_style",mLineStyle);
    par_model->GetColumn_byName<int>("color"     ,mColor    );

    const int NM = mLabel.size();
    TGraph **gm = new TGraph*[NM];

    double mSize = 1.2;
    int mMarker=20;

    for(int i=0;i<NM;i++){
       // load model
       sprintf(inpath_model,"./models/models_A1n/%s",mPath[i].c_str());
       model->ReadFile(inpath_model,true);
       // create plot
       if(mLabel[i].compare("DSE (realistic)")==0){
      mMarker = 46;
          mSize   = 2;
       }else if(mLabel[i].compare("DSE (contact)")==0){
      mMarker = 28;
          mSize   = 2;
       }else if(mLabel[i].compare("Cheng et al.")==0){
      mMarker = 42;
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
          gm[i]->SetFillColorAlpha(mColor[i],0.25); // make fill color translucent
          gm[i]->SetLineColorAlpha(mColor[i],0.00); // make line transparent
          mg->Add(gm[i],"f");                       // draw as a filled object
          LM->AddEntry(gm[i],label,"f");
       }else if(mLabel[i].compare("AdS/CFT")==0){
          gm[i]->SetFillColorAlpha(mColor[i],0.25); // make fill color translucent
          gm[i]->SetLineColorAlpha(mColor[i],0.00); // make line transparent
          mg->Add(gm[i],"f");                       // draw as a filled object
          LM->AddEntry(gm[i],label,"f");
       }else if(mLabel[i].compare("Upper AdS/CFT")==0){
          gm[i]->SetLineColorAlpha(mColor[i],1.); // make line transparent
          mg->Add(gm[i],"l");
       }else if(mLabel[i].compare("Lower AdS/CFT")==0){
          gm[i]->SetLineColorAlpha(mColor[i],1.); // make line transparent
          mg->Add(gm[i],"l");
       }else if(mLabel[i].compare("DSE (realistic)")==0){
          mg->Add(gm[i],"p");
          LM->AddEntry(gm[i],label,"p");
       }else if(mLabel[i].compare("DSE (contact)")==0){
          mg->Add(gm[i],"p");
          LM->AddEntry(gm[i],label,"p");
       }else if(mLabel[i].compare("Cheng et al.")==0){
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
    par_data->ReadFile("./input/input_exp/exp-list-A1n.csv",true);

    std::vector<std::string> exp;
    par_data->GetColumn_byName_str("exp",exp);

    std::vector<int> marker,color;
    par_data->GetColumn_byName<int>("marker",marker);
    par_data->GetColumn_byName<int>("color" ,color);

    const int N = exp.size();
    TGraphErrors **g = new TGraphErrors*[N];

    mSize = 1.2;
  
    // loop over experiments and add to plots
    for(int i=0;i<N;i++){
       // load data from a given experiment
       sprintf(inpath_data,"./data/data_A1n/%s.dat",exp[i].c_str());
       data->ReadFile(inpath_data,hasHeader);
       // create the graph
       g[i] = GetTGraphErrors(exp[i],data);
       graph_df::SetParameters(g[i],marker[i],color[i],mSize);
       // add to multigraph object
       mg->Add(g[i],"p");
       // create a label for the legend
       label = Form("%s",exp[i].c_str());
       if(exp[i].compare("JLab_E12-06-110")==0) label = Form("JLab_E12-06-110 (Proj.)");
       if(exp[i].compare("JLab_E22")==0)        label = Form("JLab 22 GeV (Proj.)");
       if(exp[i].compare("JLab_E12-06-110_no-W")!=0) L->AddEntry(g[i],label,"p");
       // set up for next data set
       data->ClearData();
    }
    
    // read in projections
    par_proj->ReadFile("./input/input_proj/proj_list/proj-list-A1n.csv",true);

    std::vector<std::string> projection;
     par_proj->GetColumn_byName_str("projection",projection);

    std::vector<int> projLine,projColor;
    par_proj->GetColumn_byName<int>("line_style",projLine);
    par_proj->GetColumn_byName<int>("color" ,projColor);

    const int NP = projection.size();
    TGraphErrors **pg = new TGraphErrors*[N];
  
    // loop over projections and add to plots
    for(int i=0;i<NP;i++){
       // load data from a given projection
       sprintf(inpath_proj,"./projections/projections_A1n/%s.dat",projection[i].c_str());
       proj->ReadFile(inpath_proj,true);
       // create the graph
       pg[i] = GetTGraphErrors(projection[i],proj);
       graph_df::SetParameters(pg[i],projLine[i],projColor[i],mSize);
       // add to multigraph object
       mg->Add(pg[i],"p");
       // create a label for the legend
       label = Form("%s (Proj.)",projection[i].c_str());
       L->AddEntry(pg[i],label,"p");
       // set up for next data set
       proj->ClearData();
    }
    

    // make the plot
    TString xAxisTitle = Form("x");
    TString yAxisTitle = Form("A_{1}^{n}");

    double xMin = 0;
    double xMax = 1;
    double yMin = -0.6;
    double yMax = 1.2;
  
    TLine *xAxisLine = new TLine(xMin,0,xMax,0);

    TCanvas *c2 = new TCanvas("c2","A1n Data",1000,800);
    c2->cd();
    
    mg->Draw("a");
    graph_df::SetLabels(mg,"",xAxisTitle,yAxisTitle);
    mg->GetXaxis()->SetLimits(xMin,xMax);
    mg->GetYaxis()->SetRangeUser(yMin,yMax);
    mg->Draw("a");
    xAxisLine->Draw("same");
    L->Draw("same");
    LM->Draw("same");
    c2->Update();

    // delete CSV objects
    delete model;
    delete data;
    delete proj;
    
    delete par_model;
    delete par_data;
    delete par_proj;
}
//______________________________________________________________________________
void PlotDQoQ(){
    // for plotting
    bool hasHeader = false;
    CSVManager *model= new CSVManager("tsv");
    CSVManager *data = new CSVManager("tsv");
    CSVManager *proj = new CSVManager("tsv");
    
    CSVManager *par_model = new CSVManager("csv");
    CSVManager *par_data = new CSVManager("csv");
    CSVManager *par_proj = new CSVManager("csv");
    
    TString label;
    char inpath_model[200];
    char inpath_data[200];
    char inpath_proj[200];
    
    TMultiGraph *mg = new TMultiGraph();
    TLegend *L      = new TLegend(0.6,0.6,0.8,0.8);
    TLegend *LM     = new TLegend(0.6,0.6,0.8,0.8);
    
    L->SetBorderSize(0);
    LM->SetBorderSize(0);
    
    // load models
    par_model->ReadFile("./input/input_model/model-list-dqoq.csv",true);
    
    std::vector<std::string> mLabel,mPath;
    par_model->GetColumn_byName_str("label",mLabel);
    par_model->GetColumn_byName_str("path" ,mPath );
    
    std::vector<int> mLineStyle,mLineWidth,mFillStyle,mColor;
    par_model->GetColumn_byName<int>("line_style",mLineStyle);
    par_model->GetColumn_byName<int>("line_width",mLineWidth);
    par_model->GetColumn_byName<int>("fill_style",mFillStyle);
    par_model->GetColumn_byName<int>("color",mColor);
    
    const int NM = mLabel.size();
        
    double mSize = 0.2;
    
    for(int i=0;i<NM;i++){
        // load model
        
        TGraph *up = new TGraph();
        TGraph *down = new TGraph();
        
        label = Form("%s",mLabel[i].c_str());
        if(mLabel[i].compare("Cheng et al.")==0){
        	sprintf(inpath_model,"./models/models_dqoq/%s",mPath[i].c_str());
       		model->ReadFile(inpath_model,true);
        	std::cout << "CHENG ET AL\n";
      		int mMarker = 42;
          	double mSize   = 2;
           	TGraph *cheng = new TGraph;
                cheng = GetModelTGraph(model);
       		graph_df::SetParameters(cheng,mMarker,mColor[i],mSize,3);
       		cheng->SetLineStyle(mLineStyle[i]);
                mg->Add(cheng,"p");
          	LM->AddEntry(cheng,label,"p");
       }else if(mLabel[i].compare("BBS+OAM")==0) {
            int nPoints = 1000;
            double stepSize = 1./((double)nPoints);
            up->AddPoint(0,0);
            up->AddPoint(0,0);
            for (int j=1; j<nPoints; j++) up->AddPoint(j*stepSize, upBBSOAM_func(j*stepSize));
            for (int j=1; j<nPoints; j++) down->AddPoint(j*stepSize, downBBSOAM_func(j*stepSize));
            up->AddPoint(1,1);
            up->AddPoint(1,1);
            
            graph_df::SetParameters(up,mLineStyle[i],mColor[i],mLineWidth[i]);
            graph_df::SetParameters(down,mLineStyle[i],mColor[i],mLineWidth[i]);
            
            mg->Add(up,"l");
            mg->Add(down,"l");
            
            LM->AddEntry(up,label,"l");
        } else {
            sprintf(inpath_model,"./models/models_dqoq/%s",mPath[i].c_str());
            model->ReadFile(inpath_model,true);
            // create plot
            
            if (mFillStyle[i]>=0){
                up = GetUpModelTGraphFill(model);
                down = GetDownModelTGraphFill(model);
                
                graph_df::SetParameters(up,mLineStyle[i],mFillStyle[i],mColor[i],mLineWidth[i]);
                graph_df::SetParameters(down,mLineStyle[i],mFillStyle[i],mColor[i],mLineWidth[i]);
                
                mg->Add(up,"lf");
                mg->Add(down,"lf");
                
                LM->AddEntry(up,label,"f");
            } else{
                up = GetUpModelTGraphLine(model);
                down = GetDownModelTGraphLine(model);
                
                graph_df::SetParameters(up,mLineStyle[i],mColor[i],mLineWidth[i]);
                graph_df::SetParameters(down,mLineStyle[i],mColor[i],mLineWidth[i]);
                
                mg->Add(up,"l");
                mg->Add(down,"l");
                LM->AddEntry(up,label,"l");
            }
        }

        // add to multigraph object
        model->ClearData();
    }
    

    
    // read in experiment names
    par_data->ReadFile("./input/input_exp/exp-list-dqoq.csv",true);
    
    std::vector<std::string> exp;
    par_data->GetColumn_byName_str("exp",exp);
    
    std::vector<int> marker,color;
    par_data->GetColumn_byName<int>("marker",marker);
    par_data->GetColumn_byName<int>("color",color);
    
    const int N = exp.size();
    TGraphErrors **u = new TGraphErrors*[N];
    TGraphErrors **d = new TGraphErrors*[N];
    
    
    mSize = 1.2;

    // loop over experiments and add to plots
    for(int i=0;i<N;i++){
        // load data from a given experiment
        sprintf(inpath_data,"./data/data_dqoq/%s.dat",exp[i].c_str());
        data->ReadFile(inpath_data,hasHeader);
        // create the graph
        u[i] = GetUpTGraphErrors(exp[i],data,false);
        d[i] = GetDownTGraphErrors(exp[i],data,false);
        graph_df::SetParameters(u[i],marker[i],color[i],mSize);
        graph_df::SetParameters(d[i],marker[i],color[i],mSize);
        // add to multigraph object
        mg->Add(u[i],"p");
        mg->Add(d[i],"p");
        // create and add label to the legend
        label = Form("%s",exp[i].c_str());
        L->AddEntry(u[i],label,"p");
        // set up for next data set
        data->ClearData();
    }
    
     // read in projections
    par_proj->ReadFile("./input/input_proj/proj_list/proj-list-dqoq.csv",true);

    std::vector<std::string> projection;
    par_proj->GetColumn_byName_str("projection",projection);

    std::vector<int> projMarker,projColor;
    par_proj->GetColumn_byName<int>("marker",projMarker);
    par_proj->GetColumn_byName<int>("color",projColor);

    const int NP = projection.size();
    TGraphErrors **up = new TGraphErrors*[NP];
    TGraphErrors **dp = new TGraphErrors*[NP];
    
    TGraph **up_pdf = new TGraph*[NP];
    TGraph **dp_pdf = new TGraph*[NP];
  
   // loop over projections and add to plots
    for(int i=0;i<NP;i++){
       // load data from a given projection
       sprintf(inpath_proj,"./projections/projections_dqoq/%s.dat",projection[i].c_str());
       proj->ReadFile(inpath_proj,true);
       // create the graph
       up[i] = GetUpTGraphErrors(projection[i],proj,true);
       dp[i] = GetDownTGraphErrors(projection[i],proj,true);
        up_pdf[i] = GetUpPDFError(proj);
        dp_pdf[i] = GetDownPDFError(proj);
       graph_df::SetParameters(up[i],projMarker[i],projColor[i],mSize);
       graph_df::SetParameters(dp[i],projMarker[i],projColor[i],mSize);
        graph_df::SetParameters(up_pdf[i],1,1001,projColor[i],mSize);
        graph_df::SetParameters(dp_pdf[i],1,1001,projColor[i],mSize);
       // add to multigraph object
       mg->Add(up[i],"p");
       mg->Add(dp[i],"p");
       // create a label for the legend
       label = Form("%s (Proj.)",projection[i].c_str());
       L->AddEntry(up[i],label,"p");
        L->AddEntry(up_pdf[i],"Projection PDF uncertainty","f");
        
        mg->Add(up_pdf[i],"f");
        mg->Add(dp_pdf[i],"f");
        
       // set up for next data set
       proj->ClearData();
    }
    
    
    // make the plot
    TString xAxisTitle = Form("x");
    TString yAxisTitle = Form("#Delta q(x)/q(x)");
    
    double xMin = 0;
    double xMax = 1;
    double yMin = -1;
    double yMax = 1;
    
    TLine *xAxisLine = new TLine(xMin,0,xMax,0);
    
    TCanvas *c3 = new TCanvas("c3","#Delta_q(x)/q(x) Data",1000,800);
    c3->cd();
    
    mg->Draw("a");
    graph_df::SetLabels(mg,"",xAxisTitle,yAxisTitle);
    mg->GetXaxis()->SetLimits(xMin,xMax);
    mg->GetYaxis()->SetRangeUser(yMin,yMax);
    mg->Draw("a");
    xAxisLine->Draw("same");
    L->Draw("same");
    LM->Draw("same");
    c3->Update();
    
    // delete CSV objects
    delete model;
    delete data;
    delete proj;
    delete par_model;
    delete par_data;
    delete par_proj;
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
TGraph *GetUpModelTGraphFill(CSVManager *model){
    // get a TGraph object for the up quark from the model data
    std::vector<double> x,u_lower,u_upper;
    model->GetColumn_byIndex<double>(0,x);
    model->GetColumn_byIndex<double>(1,u_lower);
    model->GetColumn_byIndex<double>(2,u_upper);
    
    const int N = x.size();
    TGraph *up_model = new TGraph(2*N);
    
    for (int i=0;i<N;i++) {
       up_model->SetPoint(i,x[i],u_upper[i]);
       up_model->SetPoint(N+i,x[N-i-1],u_lower[N-i-1]);
    }
    
    return up_model;
}
//______________________________________________________________________________
TGraph *GetDownModelTGraphFill(CSVManager *model){
    // get a TGraph object for the down quark from the model data
    std::vector<double> x,d_lower,d_upper;
    model->GetColumn_byIndex<double>(0,x);
    model->GetColumn_byIndex<double>(3,d_lower);
    model->GetColumn_byIndex<double>(4,d_upper);
    
    const int N = x.size();
    TGraph *down_model = new TGraph(2*N);
    
    for (int i=0;i<N;i++) {
       down_model->SetPoint(i,x[i],d_upper[i]);
       down_model->SetPoint(N+i,x[N-i-1],d_lower[N-i-1]);
    }
    
    return down_model;
}
//______________________________________________________________________________
TGraph *GetUpModelTGraphLine(CSVManager *model){
    // get a TGraph object for the up quark from the model data
    std::vector<double> x,u;
    model->GetColumn_byIndex<double>(0,x);
    model->GetColumn_byIndex<double>(1,u);
    
    TGraph *g = graph_df::GetTGraph(x,u);
    
    return g;
}
//______________________________________________________________________________
TGraph *GetDownModelTGraphLine(CSVManager *model){
    // get a TGraph object for the down quark from the model data
    std::vector<double> x,d;
    model->GetColumn_byIndex<double>(0,x);
    model->GetColumn_byIndex<double>(2,d);
    
    TGraph *g = graph_df::GetTGraph(x,d);
    
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
    }
    
    TGraphErrors *g = graph_df::GetTGraphErrors(x,ex,y,ey);
    return g;
}
//______________________________________________________________________________
TGraphErrors *GetUpTGraphErrors(std::string exp,CSVManager *data, bool ifProj){
    std::vector<double> x,ex,du_u,statdu_u,pdfdu_u,systdu_u;
    
    if(ifProj || exp.compare("JLab_E06014")==0 || exp.compare("JLab_E99117")==0 || exp.compare("JLab_EG1b")==0){
        data->GetColumn_byIndex<double>(0,x);
        data->GetColumn_byIndex<double>(2,du_u);
        data->GetColumn_byIndex<double>(3,statdu_u);
    } else {
        data->GetColumn_byIndex<double>(0,x);
        data->GetColumn_byIndex<double>(1,du_u);
        data->GetColumn_byIndex<double>(2,statdu_u);
    }
    const int N = x.size();
    for(int i=0;i<N;i++) ex.push_back(0);
    
    TGraphErrors *up_plot = graph_df::GetTGraphErrors(x,ex,du_u,statdu_u);
    
    return up_plot;
}
//______________________________________________________________________________
TGraphErrors *GetDownTGraphErrors(std::string exp,CSVManager *data, bool ifProj){
    std::vector<double> x,ex,dd_d,statdd_d;
    
    if (ifProj){
        data->GetColumn_byIndex<double>(0,x);
        data->GetColumn_byIndex<double>(6,dd_d);
        data->GetColumn_byIndex<double>(7,statdd_d);
    } else if(exp.compare("JLab_E06014")==0 || exp.compare("JLab_E99117")==0 || exp.compare("JLab_EG1b")==0) {
        data->GetColumn_byIndex<double>(0,x);
        data->GetColumn_byIndex<double>(4,dd_d);
        data->GetColumn_byIndex<double>(5,statdd_d);
    } else {
        data->GetColumn_byIndex<double>(0,x);
        data->GetColumn_byIndex<double>(3,dd_d);
        data->GetColumn_byIndex<double>(4,statdd_d);
    }
    const int N = x.size();
    for(int i=0;i<N;i++) ex.push_back(0);
    
    TGraphErrors *down_plot = graph_df::GetTGraphErrors(x,ex,dd_d,statdd_d);
    
    return down_plot;
}
//______________________________________________________________________________
TGraph *GetUpPDFError(CSVManager *data){
    std::vector<double> x,pdfdu_u,base,top;
    data->GetColumn_byIndex<double>(0,x);
    data->GetColumn_byIndex<double>(4,pdfdu_u);
    
    int n = x.size();
    
    for(int k=0;k<n;k++){
        base.push_back(-0.65);
        top.push_back(-0.65+pdfdu_u[k]);
    }
    
    TGraph *g = new TGraph(2*n);
    
    for (int k=0;k<n;k++) {
        g->SetPoint(k,x[k],top[k]);
        g->SetPoint(n+k,x[n-k-1],base[n-k-1]);
     }
    return g;
}
//______________________________________________________________________________
TGraph *GetDownPDFError(CSVManager *data){
    std::vector<double> x,pdfdd_d,base,top;
    data->GetColumn_byIndex<double>(0,x);
    data->GetColumn_byIndex<double>(8,pdfdd_d);
    
    int n = x.size();
    
    for(int k=0;k<n;k++){
        base.push_back(-0.85);
        top.push_back(-0.85+pdfdd_d[k]);
    }
    
    TGraph *g = new TGraph(2*n);
    
    for (int k=0;k<n;k++) {
        g->SetPoint(k,x[k],top[k]);
        g->SetPoint(n+k,x[n-k-1],base[n-k-1]);
     }
    return g;
}
//______________________________________________________________________________
double upBBSOAM_func(double x){
    //Formula, C, and Cprime values from Avakian, Brodsky, Deur, and Yuan 2018
    // A,B,D constant values from Brodsky, Burkardt, and Schmidt 1994
    const double Au = 3.784;
    const double Bu = -3.672;
    const double Cu = 2.004;
    const double Cprimeu= 0.493;
    const double Du = -1.892;
    const double alpha = 1.12;
    
    double arg = 1-x;
    
    double u_plus = (1./pow(x,alpha))*( (Au*pow(arg,3)) + (Bu*pow(arg,4)) );
    double u_minus = (1./pow(x,alpha))*( (Cu*pow(arg,5)) + (Cprimeu*pow(arg,5)*log10(arg)*log10(arg)) + (Du*pow(arg,6)) );
    
    double duou = (u_plus - u_minus)/(u_plus + u_minus);
    
    return duou;
}
//______________________________________________________________________________
double downBBSOAM_func(double x){
    //Formula, C, and Cprime values from Avakian, Brodsky, Deur, and Yuan 2018
    // A,B,D constant values from Brodsky, Burkardt, and Schmidt 1994 s
    const double Ad = 0.757;
    const double Bd = -0.645;
    const double Cd = 3.230;
    const double Cprimed= 1.592;
    const double Dd = -3.118;
    const double alpha = 1.12;
    
    double arg = 1.-x;
    
    double d_plus = (1./pow(x,alpha))*( (Ad*pow(arg,3)) + (Bd*pow(arg,4)) );
    double d_minus = (1./pow(x,alpha))*( (Cd*pow(arg,5)) + (Cprimed*pow(arg,5)*log10(arg)*log10(arg)) + (Dd*pow(arg,6)) );
    
    double ddod = (d_plus - d_minus)/(d_plus + d_minus);
    
    return ddod;
}

