// [[Rcpp::depends(RcppArmadillo)]]
#include<RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
RcppExport SEXP utilcomp15sigcpp_edit(SEXP y, SEXP mu, SEXP vv, SEXP frho) {

Rcpp::NumericMatrix yr(y);
Rcpp::NumericMatrix mur(mu);
Rcpp::NumericMatrix vvr(vv);
Rcpp::NumericVector frhor(frho);

int B = yr.nrow(), n = yr.ncol();

arma::mat Y(yr.begin(), B, n, false);
arma::mat MU(mur.begin(), B, n, false);
arma::mat VV(vvr.begin(), B, n, false);
arma::vec FRHO(frhor.begin(), B, false);

arma::vec ANS = arma::zeros(B);
arma::vec LL_VEC = arma::zeros(B);
arma::rowvec temp1(n);
arma::rowvec temp2(n);
arma::rowvec temp3(n);
double LL;
double LL_max;
double harris;
harris = arma::datum::pi;
harris *= 2;
harris = log(harris);
harris *= 0.5*n;

for (int i=0; i<B; i++){
  temp1 = Y.row(i);
  LL_max = -1 * arma::datum::inf;
  for (int j=0; j<B; j++){
    temp2 = temp1;
    temp2 -= MU.row(j);
    //temp2 = temp2%temp2;
    temp3 = temp2;
    temp2 %= temp3; 
    temp2 /= VV.row(j);
    LL = sum(temp2);
    LL += FRHO(j);
    LL *= -0.5;
    LL -= harris;
    LL_VEC(j) = LL;
  }
  LL_max = max(LL_VEC);
  ANS(i) = LL_max + log(mean(exp(LL_VEC-LL_max)));
}
return as<NumericVector>(wrap(ANS));
}
