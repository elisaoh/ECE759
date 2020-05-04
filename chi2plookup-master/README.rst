chi2plookup
===========

.. image:: https://travis-ci.org/MoseleyBioinformaticsLab/chi2plookup.svg?branch=master
    :target: https://travis-ci.org/MoseleyBioinformaticsLab/chi2plookup


The `chi2plookup` package provides a simple interface for creating
C++ header file for use in C++ projects. This header file contains
pregenerated array(s) of p-values for chi-square distribution for
specified degrees of freedom.

Why?
====

Need a way to calculate p-value for different degrees of freedom for a given chi2 value
and bypass third-party dependencies:

   * boost_
   * gsl_ (GNU Scientific Library)

Inspired by:

   * http://rmflight.github.io/posts/2013/10/precalcLookup.html
   * https://stackoverflow.com/questions/795972/chi-squared-probability-function-in-c

Usage example
=============

1. To view command-line help message:

.. code-block:: none

   $ python3 -m chi2plookup --help

   chi2plookup command-line interface

   Usage:
       chi2plookup -h | --help
       chi2plookup --version
       chi2plookup generate [--headerfile=<path>] [--precision=<precision>] [--df=<df>] [--start_chi=<start_chi>] [--verbose]

   Options:
       -h, --help                    Show this screen.
       --version                     Show version.
       --verbose                     Print more information.
       --headerfile=<path>           Path where to save generated header file [default: Chi2PLookup.h]
       --precision=<precision>       Parameter that controls precision [default: 10000].
       --df=<df>                     Degrees of freedom, how many to ganarate [default: 6].
       --start_chi=<start_chi>       Maximum chi2 value for given degree of freedom [default: 25].


2. Generate a header file with p-values for necessary degrees of freedom (we are using default
   number degrees of freedom, precision, and header file path):

.. code-block:: none

   $ python3 -m chi2plookup generate --verbose

   Generating p-value arrays...
     df=6
     precision=10000

       Adding p-values array to template for degree of freedom = 1 ...
       Adding p-values array to template for degree of freedom = 2 ...
       Adding p-values array to template for degree of freedom = 3 ...
       Adding p-values array to template for degree of freedom = 4 ...
       Adding p-values array to template for degree of freedom = 5 ...
       Adding p-values array to template for degree of freedom = 6 ...

3. Use generated file within your C++ project.

.. code-block:: c++

   #include <iostream>
   #include "Chi2PLookup.h"

   int main() {

       Chi2PLookup Chi2PLookupTable;
       double x = 1;
       int df = 1;
       double outvalue;

       outvalue = Chi2PLookupTable.getPValue(x, df);
       std::cout << outvalue << "\n";

       return 0;
   }


.. note:: Use the following approach for smaller number of degrees of freedom
          to avoid generating huge header files (e.g. header file with 6 degrees
          of freedom ~34 MB).


.. _boost: http://www.boost.org/doc/libs/1_65_1/libs/math/doc/html/math_toolkit/dist_ref/dists/chi_squared_dist.html
.. _gsl: http://www.gnu.org/software/gsl/doc/html/randist.html?highlight=chi#the-chi-squared-distribution
