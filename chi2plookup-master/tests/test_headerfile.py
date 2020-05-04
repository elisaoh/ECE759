#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__  import print_function, division
import subprocess
from scipy.stats import chi2


TESTFILE_TEMPLATE = """#include <iostream>
#include "Chi2PLookup.h"

int main() {{

    Chi2PLookup Chi2PLookupTable;
    double x = {0};
    int df = {1};
    double outvalue;

    outvalue = Chi2PLookupTable.getPValue(x, df);
    std::cout << outvalue << "\\n";

    return 0;
}}
"""


def test_headerfile(template=TESTFILE_TEMPLATE, testvalue=1.1,
                    df=1, precision=10000, start_chi=25, headerfile="tests/Chi2PLookup.h",
                    srcfpath="tests/test.cpp", binfpath="tests/test.out"):
    """Test generated header file within cpp source file.

    :param str template: Template file that contains main() function and imports header file.
    :param testvalue: Chi value.
    :param int df: Degree of freedom.
    :param str srcfpath: Path where source file will be saved.
    :param str binfpath: Path where binary file will be saved.
    :return: None
    :rtype: None
    """
    command = "python -m chi2plookup generate --headerfile={} --df={} --precision={} --start_chi={}".format(headerfile, df, precision, start_chi)
    subprocess.call(command, shell=True)
    p_value = 1 - chi2.cdf(testvalue, df)

    template = template.format(testvalue, df)
    with open(srcfpath, "w") as outfile:
        outfile.write(template)

    subprocess.call("g++ -std=c++11 {} -o {}".format(srcfpath, binfpath), shell=True)
    generated_p_value = subprocess.check_output("./{}".format(binfpath))

    assert round(float(p_value), 6) == round(float(generated_p_value.strip()), 6)
