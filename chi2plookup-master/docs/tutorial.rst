The chi2plookup Tutorial
========================

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. code::

   chi2plookup command-line interface

   Usage:
       chi2plookup -h | --help
       chi2plookup --version
       chi2plookup generate [--headerfile=<path>] [--precision=<precision>] [--df=<df>] [--start_chi=<start_chi>] [--verbose]

   Options:
       -h, --help                    Show this screen.
       --version                     Show version.
       --verbose                     Print more information.
       --headerfile=<path>           Path where to save generated header file [default: Chi2PValues.h]
       --precision=<precision>       Parameter that controls precision [default: 10000].
       --df=<df>                     Degrees of freedom, how many to ganarate [default: 6].
       --start_chi=<start_chi>       Maximum chi2 value for given degree of freedom [default: 25].