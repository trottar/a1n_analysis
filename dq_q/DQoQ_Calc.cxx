// compute du/u and dd/d from A1n, A1p and LHAPDF values

#include "LHAPDF/LHAPDF.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "./src/CSVManager.cpp"
#include "./src/Math.cpp"
using namespace LHAPDF;
using namespace std;

double GetDUoU(double A1n,double A1p,double x,double Q2);
double GetDDoD(double A1n,double A1p,double x,double Q2);

string pdfName = "CT18NLO"
const PDF* pdf = mkPDF(pdfName,0);

int DQoQ_Calc(){
    
    //read in A1 projection names
    char inpath_par[200];
    /*This is not in the input file yet*/
    sprintf(inpath_par,"./input/model-list.csv");
    CSVManager *par = new CSVManager("csv");
    par->ReadFile(inpath_par,true);
    
    std::vector<std::string> path,label;
    /* Name the column proj*/
    par->GetColumn_byName_str("path",path);
    par->GetColumn_byName_str("label",label);
    
    const int N = proj.size();
    
    CSVManager *data = new CSVManager("tsv");
    CSVManager *output = new CSVManager("tsv");
    
    for(int i=0;i<N;i++){
        char inpath[200];
        char outpath[200];
        // load data from a given projection
        sprintf(inpath,"./data/models/%s",path[i].c_str());
        data->ReadFile(inpath,hasHeader);
        //set up a file to export data; naming convention "dqoq_NameOfProjection.dat"
        sprintf(outpath,"./data/dqoq_data/dqoq_%s.dat",proj[i].c_str());
        // extracting data from the projection
        std::vector<double> x,Q2,A1p,A1n,EA1p,EA1n;
        data->GetColumn_byIndex<double>(0,x);
        data->GetColumn_byIndex<double>(1,Q2);
        data->GetColumn_byIndex<double>(2,A1p);
        data->GetColumn_byIndex<double>(3,EA1p);
        data->GetColumn_byIndex<double>(4,A1n);
        data->GetColumn_byIndex<double>(4,EA1n);
        //calculating dq/q and errors
        const int M = x.size();
        std::vector<double> du_u,Edu_u,dd_d,Edd_d;
        double arg_u =0;
        double arg_d =0;
        for(int j=0;j<M;j++) {
            //getting du/u and dd/d
            arg_u = GetDUoU(A1p[j],A1n[j],x[i],Q2[i]);
            arg_d = GetDDoD(A1p[j],A1n[j],x[i],Q2[i]);
            //adding these values to du_u and dd_d vectors
            du_u.push_back(arg_u);
            dd_d.push_back(arg_d);
            /*Error Propagation yet to be completed*/
            Edu_u.pushback(0);
            Edd_d.pushback(0);
        }
        //outputting dq/q to the new output file
        csv->InitTable(NP,5);
        csv->SetHeader("x,du/u,Edu/u,dd/d,Edd/d");
        csv->SetColumn<double>(0,x);
        csv->SetColumn<double>(1,du_u);
        csv->SetColumn<double>(2,Edu_u);
        csv->SetColumn<double>(3,dd_d);
        csv->SetColumn<double>(4,Edd_d);
        csv->WriteFile(outpath);
        //resetting for next file
        data->ClearData();
    }

    return 0;
}
//______________________________________________________________________________
double GetDUoU(double A1p,double A1n,double d_u){
    //  calculating du/u
    double du_u=( (((4/15)*A1p)*(4 + (d_u))) - (((1/15)*A1n)*(1 + 4*(d_u))) );
    return du_u;
}
//______________________________________________________________________________
double GetDDoD(double A1p,double A1n,double d_u){
    //  calculating dd/d
    double dd_d=( (((4/15)*A1n)*(4 + (d_u))) - (((1/15)*A1p)*(1 + 4*(d_u))) );
    return dd_d;
}
//______________________________________________________________________________
double GetDUoUError(double A1p,double EA1p,double A1n,double EA1n,double d_u,double Ed_u){
    //  calculating error in du/u
    double left,right,Edu_u;
    left = (4/15) * ( (A1p) * (4 + (d_u)) ) * sqrt( ((EA1p/A1p)*(EA1p/A1p)) + ((Ed_u/(d_u+4))*(Ed_u/(d_u+4))));
    right = (1/15) * ( (A1n) * (1 + 4*(d_u)) ) * sqrt( ((EA1n/A1n)*(EA1n/A1n)) + ((4*Ed_u/(4*d_u+1))*(4*Ed_u/(4*d_u+1))));
    double Edu_u= sqrt ( (left*left) + (right*right) );
    return Edu_u;
}
//______________________________________________________________________________
double GetDDoDError(double A1p,double EA1p,double A1n,double EA1n,double d_u,double Ed_u){
    //  calculating error in dd/d
    double left,right,Edd_d;
    double dd_d=( (((4/15)*A1n)*(4 + ((1/d_u))) - (((1/15)*A1p)*(1 + (4/(d_u)))) ) );
    return dd_d;
}
//______________________________________________________________________________
double GetPDFError(string pdf_set,double x,double Q2){
     //  calculating error in dd/d
    double delta_O;
    double sum = 0;
    double diff;
    int PDF_size; //define this
    
    if (pdf_set.find("CT18NLO") != std::string::npos) {
        PDF_size = 59;
    } else if (pdf_set.find("NNPDF") != std::string::npos) {
        PDF_size = 101;
    }
    
    if (pdf_set.find("CT18NLO") != std::string::npos || pdf_set.find("MMHT20") != std::string::npos || pdf_set.find("PDF4LHC") != std::string::npos) {
        //Hessian-based set
        for (int i = 1; i<=PDF_size/2; i++) {
            const PDF* pdf1 = mkPDF(pdf_set, 2i);
            const PDF* pdf2 = mkPDF(pdf_set, 2i-1);
            diff = ( (pdf2->xfxQ2(x,Q2)) - (pdf1->xfxQ2(x,Q2)) );
            sum += diff * diff;
        }
        delta_O = sqrt ( 0.25 * sum );
    } else {
        //replica-based set
        for (int i = 1; i<=PDF_size; i++) {
            const PDF* pdf_nom = mkPDF(pdf_set, 0);
            const PDF* pdf_m = mkPDF(pdf_set, i);
            diff = ( (pdf_m->xfxQ2(x,Q2)) - (pdf_nom->xfxQ2(x,Q2)) );
            sum += diff * diff;
        }
        delta_O = sqrt ( (1/PDF_size) * sum );
    }
    return delta_O;
}
