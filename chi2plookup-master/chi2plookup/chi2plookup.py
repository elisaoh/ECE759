#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__  import print_function, division
import os
from scipy.stats import chi2


HEADERFILE_TEMPLATE = """#ifndef CHI2PLOOKUP_H
#define CHI2PLOOKUP_H


struct Chi2PLookup
{{
    static const double * pValues[];
    static const int cutoff[];
    static const int divisor;

    inline double getPValue(double statistic, int df) {{
        return((statistic >= cutoff[df-1]) ? 0.0 : pValues[df-1][int(divisor * statistic)]);
    }}
    
}};

{0}
{1}

{2}
{3}

#endif // CHI2PLOOKUP_H
"""


def max_chi_value(df=1, start_chi=25):
    """Determine maximum chi2 value statistic that can be used 
    for a given degree of freedom. See chi2 distribution graph 
    that shifts as the number of degrees of freedom increases
    and why we need to adjust max_chi parameter 
    (https://en.wikipedia.org/wiki/Chi-squared_distribution)

    :param int df: Degree of freedom.
    :param int start_chi: Maximum chi2 value for a given degree of freedom.
    :return: Maximum chi value statistic for a given degree of freedom.
    :rtype: int
    """
    if df == 1:
        return start_chi

    start_p_value = 1 - chi2.cdf(start_chi, 1)
    max_chi = start_chi
    p_value = 1 - chi2.cdf(max_chi, df)

    while p_value >= start_p_value:
        max_chi += 1
        p_value = 1 - chi2.cdf(max_chi, df)

    return max_chi


def generate_headerfile(template, n_division=10000, df=6, start_chi=25, filepath="Chi2PLookup.h", verbose=False):
    """Generate C++ header file that contain pre-generated array of arrays of p-values for specified
    degrees of freedom.

    :param str template: Header file template.
    :param int n_division: Precision.
    :param int df: Degrees of freedom.
    :param int start_chi: Maximum chi value for degree of freedom = 1.
    :param str filepath: Path where header file will be saved.
    :return: String containing 
    :rtype: :py:class:`str`
    """
    divisor = "const int Chi2PLookup::divisor = {};".format(n_division)

    names = []
    cutoff = []
    p_values_arrays = []
    degrees_of_freedom = range(1, df+1)

    if verbose:
        print("Generating p-value arrays...")
        print("  df={}".format(df))
        print("  precision={}".format(n_division))

    for df in degrees_of_freedom:
        var_name = "pValues_{}".format(df)
        names.append(var_name)
        max_chi = max_chi_value(df=df, start_chi=start_chi)
        cutoff.append(max_chi)
        n_elements = max_chi * n_division

        chi_values = (val / n_division for val in range(0, n_elements + 1))
        p_values = (str(1 - chi2.cdf(val, df)) for val in chi_values)

        if verbose:
            print("\tAdding p-values array to template for degree of freedom = {} ...".format(df))

        p_values_arrays.append("double {}[] = {{{}}};".format(var_name, ", ".join(p_values)))

    cutoff_array = "const int Chi2PLookup::cutoff[] = {{{}}};".format(", ".join([str(i) for i in cutoff]))
    p_values_array_of_arrays = "const double * Chi2PLookup::pValues[] = {{{}}};\n".format(", ".join(names))

    template = template.format(divisor, cutoff_array, "\n".join(p_values_arrays), p_values_array_of_arrays)

    if verbose:
        print("Saving file to: {}".format(os.path.abspath(filepath)))

    with open(filepath, "w") as outfile:
        outfile.write(template)

    return template
