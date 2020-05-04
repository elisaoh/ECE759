#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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
"""

from . import chi2plookup


def cli(cmdargs):
    if cmdargs["generate"]:
        chi2plookup.generate_headerfile(template=chi2plookup.HEADERFILE_TEMPLATE,
                                        n_division=int(cmdargs["--precision"]),
                                        df=int(cmdargs["--df"]),
                                        start_chi=int(cmdargs["--start_chi"]),
                                        filepath=cmdargs["--headerfile"],
                                        verbose=cmdargs["--verbose"])
