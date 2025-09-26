#!/usr/bin/env python3
"""
Simple wrapper to run the surgical VOP app from root directory for deployment.
"""
import sys
import os

# Add the surgical-vop-assessment directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from surgical_vop_app import *

if __name__ == "__main__":
    # The app will run when this file is executed
    pass