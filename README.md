# The model files are too large to be included in the GitHub repository, If you need to download the complete model files, please visit the following link: https://drive.google.com/drive/folders/1oNXUASF-3Jg4JHvnXDT57VvK-MIIeGbx?usp=sharing

# then follow the Readme.md steps to run the project

# How to Run the Project

This is a frontend–backend separated project. Please follow the steps below to run it locally.

# 1. Enable PowerShell Script Execution (First-Time Setup Only)

Open the Start Menu, search for PowerShell

Right-click → Run as Administrator

Enter the following command:

Set-ExecutionPolicy RemoteSigned


Type Y to confirm

Close PowerShell


# 2. Start the Backend

Open your terminal in the project directory

Activate the virtual environment:

.\.venv\Scripts\activate


Run the backend server:

python -m uvicorn app:app --host 127.0.0.1 --port 8000


The backend will now be running at:
http://127.0.0.1:8000

# 3. Start the Frontend

In a separate terminal window:

npm run dev
