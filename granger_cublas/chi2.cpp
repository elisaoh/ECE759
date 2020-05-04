//g++ chi2test.cpp -o chi2test


#include <iostream>
#include "Chi2PLookup.h"


double chi2_Pvalue(double ssr_own, double ssr_joint, int nobs, int df){
Chi2PLookup Chi2PLookupTable;
double x = nobs*(ssr_own - ssr_joint) / ssr_joint;

return Chi2PLookupTable.getPValue(x, df);
}

// int main() {

//     Chi2PLookup Chi2PLookupTable;
//     res2down.nobs
// res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
//     double x = -5.811074918566775;
//     int df = 2;
//     double outvalue;

//     outvalue = Chi2PLookupTable.getPValue(x, df);
//     std::cout << 1- outvalue << "\n";

//     return 0;
// }