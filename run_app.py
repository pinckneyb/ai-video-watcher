#!/usr/bin/env python3
"""
Direct runner for surgical VOP app - avoids workflow port auto-detection
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'surgical-vop-assessment'))

# Import the app
if __name__ == "__main__":
    import subprocess
    subprocess.run([
        "streamlit", "run", 
        "surgical-vop-assessment/surgical_vop_app.py",
        "--server.port=5000",
        "--server.address=0.0.0.0"
    ])