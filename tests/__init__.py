import os
import sys

# make the in-repo package importable when the test suite is run from a
# source checkout without installing the package first
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
