// Make a plot of the model A1n data 

#include "LHAPDF/LHAPDF.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "./src/CSVManager.cpp"

using namespace LHAPDF;
using namespace std;

double GetDUoU(double x,double Q2, double A1p, double A1n, string pdfName,int setNum);
double GetDDoD(double x,double Q2, double A1p, double A1n, string pdfName,int setNum);
double GetDUoUError_stat(double x,double Q2,double A1p,double EA1p,double A1n,double EA1n,string pdfName);
double GetDDoDError_stat(double x,double Q2,double A1p,double EA1p,double A1n,double EA1n,string pdfName);
double GetDUoUError_pdf(double x,double Q2, double A1p, double A1n, string pdfName, int PDF_size,string errorType);
double GetDDoDError_pdf(double x,double Q2, double A1p, double A1n, string pdf_set, int PDF_size,string errorType);
double linearInterpolate(double xmin,double xmax,double xtarget,double ymin,double ymax);
int ProjectionA1p();
int ProjectionA1n();
int ProjectionDqoq();

int main(){
    ProjectionA1p();
    ProjectionA1n();
    ProjectionDqoq();
    
    return 0;
}
//______________________________________________________________________________
double GetDUoU(double x,double Q2, double A1p, double A1n, string pdfName,int setNum){
    //  calculating du/u
    PDF* pdf = mkPDF(pdfName, setNum);
    double d = (pdf->xfxQ2(1,x,Q2))/x;
    double u = (pdf->xfxQ2(2,x,Q2))/x;
    double d_u = d/u;
    double du_u = ( ( (4./15.)*(A1p)*(4. + d_u) ) - ( (1./15.)*(A1n)*(1. + (4.*d_u) ) ) );
    return du_u;
}
//______________________________________________________________________________
double GetDDoD(double x,double Q2, double A1p, double A1n, string pdfName,int setNum) {
    //  calculating dd/d
    PDF* pdf = mkPDF(pdfName, setNum);
    double d = (pdf->xfxQ2(1,x,Q2))/x;
    double u = (pdf->xfxQ2(2,x,Q2))/x;
    double d_u = d/u;
    double dd_d = ( ( (4./15.)*(A1n)*(4. + (1/d_u)) ) - ( (1./15.)*(A1p)*(1. + (4./d_u) ) ) );
    return dd_d;
}
//______________________________________________________________________________
double GetDUoUError_stat(double x,double Q2,double A1p,double EA1p,double A1n,double EA1n,string pdfName){
    //  calculating error in du/u
    const PDF* pdf = mkPDF(pdfName, 0);
    double d = (pdf->xfxQ2(1,x,Q2))/x;
    double u = (pdf->xfxQ2(2,x,Q2))/x;
    double d_u = d/u;
    std::cout << "x = " << x << " Q2 = " << Q2 << " d_u = " << d_u << std::endl;
    double partialA1p = (4./15.)*(4.+d_u);
    double partialA1n = (-1./15.)*(1.+(4.*d_u));
    double Edu_u= sqrt ( ((partialA1p*EA1p)*(partialA1p*EA1p)) + ((partialA1n*EA1n)*(partialA1n*EA1n)) );
    return Edu_u;
}
//______________________________________________________________________________
double GetDDoDError_stat(double x,double Q2,double A1p,double EA1p,double A1n,double EA1n,string pdfName){
    //  calculating error in du/u
    const PDF* pdf = mkPDF(pdfName, 0);
    double d = (pdf->xfxQ2(1,x,Q2))/x;
    double u = (pdf->xfxQ2(2,x,Q2))/x;
    double u_d = u/d;
    double partialA1p = (-1./15.)*(1.+(4*u_d));
    double partialA1n = (4./15.)*(4.+u_d);
    double Edd_d= sqrt ( ((partialA1p*EA1p)*(partialA1p*EA1p)) + ((partialA1n*EA1n)*(partialA1n*EA1n)) );
    return Edd_d;
}
//______________________________________________________________________________
double GetDUoUError_pdf(double x,double Q2, double A1p, double A1n, string pdf_set, int PDF_size,string errorType){
   //  calculating error in dd/d
   double delta_O;
   double sum = 0;
   double diff;

   if (errorType.compare("Hessian")==0){
       //Hessian-based set
       for (int i = 1; i<=PDF_size/2; i++) {
           double du_u1 = GetDUoU(x,Q2,A1p,A1n,pdf_set,2*i);
           double du_u2 = GetDUoU(x,Q2,A1p,A1n,pdf_set,(2*i)-1);
           diff = du_u2 - du_u1;
           sum += diff * diff;
       }
       delta_O = sqrt ( 0.25 * sum );
   } else if (errorType.compare("Replica")==0){
       //replica-based set
       for (int i = 1; i<=PDF_size; i++) {
           double du_unom = GetDUoU(x,Q2,A1p,A1n,pdf_set,0);
           double du_um = GetDUoU(x,Q2,A1p,A1n,pdf_set,i);
           diff = du_unom - du_um;
           sum += diff * diff;
           
       }
       delta_O = sqrt ( (1./PDF_size) * sum );
   }
   return delta_O;
}
//______________________________________________________________________________
double GetDDoDError_pdf(double x,double Q2, double A1p, double A1n, string pdf_set, int PDF_size,string errorType){
   //  calculating error in dd/d
   double delta_O;
   double sum = 0;
   double diff;

   if (errorType.compare("Hessian")==0) {
       //Hessian-based set
       for (int i = 1; i<=PDF_size/2; i++) {
           double dd_d1 = GetDDoD(x,Q2,A1p,A1n,pdf_set,2*i);
           double dd_d2 = GetDDoD(x,Q2,A1p,A1n,pdf_set,(2*i)-1);
           diff = dd_d2 - dd_d1;
           sum += diff * diff;
       }
       delta_O = sqrt ( 0.25 * sum );
   } else if (errorType.compare("Replica")==0){
       //replica-based set
       for (int i = 1; i<=PDF_size; i++) {
           double dd_dnom = GetDDoD(x,Q2,A1p,A1n,pdf_set,0);
           double dd_dm = GetDDoD(x,Q2,A1p,A1n,pdf_set,i);
           diff = dd_dnom - dd_dm;
           sum += diff * diff;
       }
       delta_O = sqrt ( (1./PDF_size) * sum );
   }
   return delta_O;
}
//______________________________________________________________________________
double linearInterpolate(double x1,double x2,double xtarget,double y1,double y2) {
    double slope = (y2-y1)/(x2-x1);
    double ytarget = ((xtarget-x1)*slope) + y1;
    return ytarget;
}
//______________________________________________________________________________
int ProjectionA1p() {
    // load models
    CSVManager *A1p_manager = new CSVManager("csv");
    A1p_manager->ReadFile("./input/input_proj/proj_build/proj-build-A1p.csv",true);
    
    std::vector<std::string> projection, model, error;
    A1p_manager->GetColumn_byName_str("projection",projection);
    A1p_manager->GetColumn_byName_str("model" ,model );
    A1p_manager->GetColumn_byName_str("error" ,error );
    
    CSVManager *A1p_model= new CSVManager("tsv");
    CSVManager *A1p_error= new CSVManager("tsv");
    CSVManager *Project_out = new CSVManager("tsv");
    
    const int NM = projection.size();
    
    char modelPath[200];
    char errorPath[200];
    char outPath[200];
    
    for(int i=0;i<NM;i++){
        // load model
        sprintf(modelPath,"./models/models_A1p/%s",model[i].c_str());
        sprintf(errorPath,"./error/error_A1p/%s",error[i].c_str());
        sprintf(outPath,"./projections/projections_A1p/%s.dat",projection[i].c_str());
        
        A1p_model->ReadFile(modelPath,true);
        A1p_error->ReadFile(errorPath,true);
        
        std::vector<double> xModel,xError,Q2,A1pModel,A1p_Interpolate,dA1p,syst;
        A1p_model->GetColumn_byIndex<double>(0,xModel);
        A1p_model->GetColumn_byIndex<double>(1,A1pModel);
        A1p_error->GetColumn_byIndex<double>(0,xError);
        A1p_error->GetColumn_byIndex<double>(1,dA1p);
        
        const int N = xError.size();
        const int M = xModel.size();
        for(int j=0;j<N;j++){
            
            int closestXModel = 0;
            
            for(int z=0;z<M;z++){
                if( abs(xError[j]-xModel[z])<abs(xError[j]-xModel[closestXModel]) ) {
                    closestXModel = z;
                }
            }
            double newA1p;
            if (xError[j]==xModel[closestXModel]) {
                newA1p = A1pModel[closestXModel];
                A1p_Interpolate.push_back(newA1p);
            } else if (xError[j]>xModel[closestXModel]) {
                if( (closestXModel+1)<xModel.size() ) {
                    newA1p = linearInterpolate(xModel[closestXModel],xModel[closestXModel+1],xError[j],A1pModel[closestXModel],A1pModel[closestXModel+1]);
                    A1p_Interpolate.push_back(newA1p);
                } else {
                    std::cout << "ERROR!  Projection is out of range of the model"<< std::endl;
                }
            } else {
                if( (closestXModel-1)>=0 ) {
                    newA1p = linearInterpolate(xModel[closestXModel-1],xModel[closestXModel],xError[j],A1pModel[closestXModel-1],A1pModel[closestXModel]);
                    A1p_Interpolate.push_back(newA1p);
                } else {
                    std::cout << "ERROR!  Projection is out of range of the model"<< std::endl;
                }
            }
            Q2.push_back(5);
            syst.push_back(0);
            
        }
        
        Project_out->InitTable(N,5);
        Project_out->SetHeader("x,Q2,A1p,stat,syst");
        Project_out->SetColumn<double>(0,xError);
        Project_out->SetColumn<double>(1,Q2);
        Project_out->SetColumn<double>(2,A1p_Interpolate);
        Project_out->SetColumn<double>(3,dA1p);
        Project_out->SetColumn<double>(4,syst);
        Project_out->WriteFile(outPath);
    }
    
    delete A1p_manager;
    delete A1p_model;
    delete A1p_error;
    delete Project_out;
    
    return 0;
}
//______________________________________________________________________________
int ProjectionA1n() {
    // load models
    CSVManager *A1n_manager = new CSVManager("csv");
    A1n_manager->ReadFile("./input/input_proj/proj_build/proj-build-A1n.csv",true);
    
    std::vector<std::string> projection, model, error;
    A1n_manager->GetColumn_byName_str("projection",projection);
    A1n_manager->GetColumn_byName_str("model" ,model );
    A1n_manager->GetColumn_byName_str("error" ,error );
    
    CSVManager *A1n_model= new CSVManager("tsv");
    CSVManager *A1n_error= new CSVManager("tsv");
    CSVManager *Project_out = new CSVManager("tsv");
    
    const int NM = projection.size();
    
    char modelPath[200];
    char errorPath[200];
    char outPath[200];
    
    for(int i=0;i<NM;i++){
        // load model
        sprintf(modelPath,"./models/models_A1n/%s",model[i].c_str());
        sprintf(errorPath,"./error/error_A1n/%s",error[i].c_str());
        sprintf(outPath,"./projections/projections_A1n/%s.dat",projection[i].c_str());
        
        A1n_model->ReadFile(modelPath,true);
        A1n_error->ReadFile(errorPath,true);
        
        std::vector<double> xModel,xError,Q2,A1nModel,A1n_Interpolate,dA1n,syst;
        A1n_model->GetColumn_byIndex<double>(0,xModel);
        A1n_model->GetColumn_byIndex<double>(1,A1nModel);
        A1n_error->GetColumn_byIndex<double>(0,xError);
        A1n_error->GetColumn_byIndex<double>(1,dA1n);
        
        const int N = xError.size();
        const int M = xModel.size();
        for(int j=0;j<N;j++){
            
            int closestXModel = 0;
            
            for(int z=0;z<M;z++){
                if( abs(xError[j]-xModel[z])<abs(xError[j]-xModel[closestXModel]) ) {
                    closestXModel = z;
                }
            }
            double newA1n;
            if (xError[j]==xModel[closestXModel]) {
                newA1n = A1nModel[closestXModel];
                A1n_Interpolate.push_back(newA1n);
            } else if (xError[j]>xModel[closestXModel]) {
                if( (closestXModel+1)<xModel.size() ) {
                    newA1n = linearInterpolate(xModel[closestXModel],xModel[closestXModel+1],xError[j],A1nModel[closestXModel],A1nModel[closestXModel+1]);
                    A1n_Interpolate.push_back(newA1n);
                } else {
                    std::cout << "ERROR!  Projection is out of range of the model"<< std::endl;
                }
            } else {
                if( (closestXModel-1)>=0 ) {
                    newA1n = linearInterpolate(xModel[closestXModel-1],xModel[closestXModel],xError[j],A1nModel[closestXModel-1],A1nModel[closestXModel]);
                    A1n_Interpolate.push_back(newA1n);
                } else {
                    std::cout << "ERROR!  Projection is out of range of the model"<< std::endl;
                }
            }
            Q2.push_back(5);
            syst.push_back(0);
            
        }
        
        Project_out->InitTable(N,5);
        Project_out->SetHeader("x,Q2,A1n,stat,syst");
        Project_out->SetColumn<double>(0,xError);
        Project_out->SetColumn<double>(1,Q2);
        Project_out->SetColumn<double>(2,A1n_Interpolate);
        Project_out->SetColumn<double>(3,dA1n);
        Project_out->SetColumn<double>(4,syst);
        Project_out->WriteFile(outPath);
    }
    
    delete A1n_manager;
    delete A1n_model;
    delete A1n_error;
    delete Project_out;
    
    return 0;
}
//______________________________________________________________________________
 int ProjectionDqoq() {
    
    string pdfName = "CJ15nlo";
    int pdfSize = 49;
    string errorType = "Hessian";
    
    // load models
    CSVManager *dqoq_manager = new CSVManager("csv");
    dqoq_manager->ReadFile("./input/input_proj/proj_build/proj-build-dqoq.csv",true);
    
    std::vector<std::string> output, A1p_proj, A1n_proj;
    dqoq_manager->GetColumn_byName_str("A1p_projection",A1p_proj);
    dqoq_manager->GetColumn_byName_str("A1n_projection",A1n_proj);
    dqoq_manager->GetColumn_byName_str("dqoq_projection",output);

    
    CSVManager *A1p_manager= new CSVManager("tsv");
    CSVManager *A1n_manager= new CSVManager("tsv");
    CSVManager *Project_out = new CSVManager("tsv");
    
    const int NM = output.size();
    
    char A1pPath[200];
    char A1nPath[200];
    char outPath[200];
    
    for(int i=0;i<NM;i++){
        // load model
        sprintf(A1pPath,"./projections/projections_A1p/%s",A1p_proj[i].c_str());
        sprintf(A1nPath,"./projections/projections_A1n/%s",A1n_proj[i].c_str());
        sprintf(outPath,"./projections/projections_dqoq/%s.dat",output[i].c_str());
        
        A1p_manager->ReadFile(A1pPath,true);
        A1n_manager->ReadFile(A1nPath,true);
        
        std::vector<double> xA1p, A1p, statA1p, xA1n, A1n, statA1n;
        A1p_manager->GetColumn_byIndex<double>(0,xA1p);
        A1p_manager->GetColumn_byIndex<double>(2,A1p);
        A1p_manager->GetColumn_byIndex<double>(3,statA1p);
        A1n_manager->GetColumn_byIndex<double>(0,xA1n);
        A1n_manager->GetColumn_byIndex<double>(2,A1n);
        A1n_manager->GetColumn_byIndex<double>(3,statA1n);
    
        std::vector<double> finalx, finalQ2, finalA1p, finalstatA1p, finalA1n, finalstatA1n;
        
        const int P = xA1p.size();
        const int N = xA1n.size();
        
        for(int j=0;j<P;j++){
            for(int k=0;k<N;k++){
                if(xA1p[j]==xA1n[k]){
                    finalx.push_back(xA1p[j]);
                    finalQ2.push_back(5);
                    finalA1p.push_back(A1p[j]);
                    finalstatA1p.push_back(statA1p[j]);
                    finalA1n.push_back(A1n[k]);
                    finalstatA1n.push_back(statA1n[k]);
                }
            }
        }
        
        std::vector<double> Q2, du_u,statdu_u,pdfdu_u,systdu_u,dd_d,statdd_d,pdfdd_d,systdd_d;
        
        const int F = finalx.size();
        
        for(int z=0;z<F;z++){
            du_u.push_back(GetDUoU(finalx[z],finalQ2[z],finalA1p[z],finalA1n[z],pdfName,0));
            dd_d.push_back(GetDDoD(finalx[z],finalQ2[z],finalA1p[z],finalA1n[z],pdfName,0));
            statdu_u.push_back(GetDUoUError_stat(finalx[z],finalQ2[z],finalA1p[z],finalstatA1p[z],finalA1n[z],finalstatA1n[z],pdfName));
            statdd_d.push_back(GetDDoDError_stat(finalx[z],finalQ2[z],finalA1p[z],finalstatA1p[z],finalA1n[z],finalstatA1n[z],pdfName));
            pdfdu_u.push_back(GetDUoUError_pdf(finalx[z],finalQ2[z],finalA1p[z],finalA1n[z],pdfName,pdfSize,errorType));
            pdfdd_d.push_back(GetDDoDError_pdf(finalx[z],finalQ2[z],finalA1p[z],finalA1n[z],pdfName,pdfSize,errorType));
            systdu_u.push_back(0);
            systdd_d.push_back(0);
            Q2.push_back(5);
        }
        
        Project_out->InitTable(F,10);
        Project_out->SetHeader("x,Q2,du_u,statdu_u,pdfdu_u,systdu_u,dd_d,statdd_d,pdfdd_d,systdd_d");
        Project_out->SetColumn<double>(0,finalx);
        Project_out->SetColumn<double>(1,Q2);
        Project_out->SetColumn<double>(2,du_u);
        Project_out->SetColumn<double>(3,statdu_u);
        Project_out->SetColumn<double>(4,pdfdu_u);
        Project_out->SetColumn<double>(5,systdu_u);
        Project_out->SetColumn<double>(6,dd_d);
        Project_out->SetColumn<double>(7,statdd_d);
        Project_out->SetColumn<double>(8,pdfdd_d);
        Project_out->SetColumn<double>(9,systdd_d);
        
        Project_out->WriteFile(outPath);
    }
    
    delete dqoq_manager;
    delete A1p_manager;
    delete A1n_manager;
    delete Project_out;
    
    return 0;
}

