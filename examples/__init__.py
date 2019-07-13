# -*- coding: utf-8 -*-
import os
import sys
os.chdir('..')
package_path = os.getcwd()
if package_path not in sys.path:
    sys.path.append(package_path)
#print(sys.path)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
