# Hackathon Setup Instructions

**IMPORTANT FOR JUDGES:** 
Please run this project using **standard Python 3.10+ from python.org** on Windows, Mac, or Linux. 
Do **NOT** use MSYS2 or MinGW Python, as PyTorch does not provide pre-compiled binary wheels for those environments, which will cause installation failures during the evaluation.

## Quick Start (Windows)
Simply double-click `start.bat`. It will automatically:
1. Create a virtual environment using `py -3.10`
2. Install the correct CPU-only PyTorch wheels and Ultralytics
3. Start the backend server and open the frontend UI

## Manual Setup (Mac/Linux)
If you are evaluating on Mac or Linux, please run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```
