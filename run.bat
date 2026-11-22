@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || goto folder_error

set "PYTHONUTF8=1"
set "VENV_PYTHON=.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto check_venv

echo Creating the Python environment...
py -3.12 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>&1
if not errorlevel 1 goto create_with_312
py -3.11 -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" >nul 2>&1
if not errorlevel 1 goto create_with_311
python -c "import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))" >nul 2>&1
if errorlevel 1 goto python_error
python -m venv .venv
goto check_creation

:create_with_312
py -3.12 -m venv .venv
goto check_creation

:create_with_311
py -3.11 -m venv .venv

:check_creation
if errorlevel 1 goto venv_error

:check_venv
"%VENV_PYTHON%" -c "import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))" >nul 2>&1
if errorlevel 1 goto venv_error
if not exist .venv\installed-requirements.txt goto install
"%VENV_PYTHON%" -c "from pathlib import Path; import sys; sys.exit(Path('requirements.txt').read_bytes() != Path('.venv/installed-requirements.txt').read_bytes())" >nul 2>&1
if not errorlevel 1 goto launch

:install
echo Installing required packages. This can take a few minutes...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto install_error
copy /y requirements.txt .venv\installed-requirements.txt >nul
if errorlevel 1 goto install_error

:launch
echo The meeting app will open at http://127.0.0.1:8501
"%VENV_PYTHON%" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
if errorlevel 1 goto launch_error
popd
exit /b 0

:python_error
echo Python 3.12 or 3.11 is required. See README.md for installation.
goto fail

:venv_error
echo Could not start the Python environment. Rename .venv and run this file again.
goto fail

:install_error
echo Package setup failed. Check the messages above and your internet connection.
goto fail

:launch_error
echo The application could not start. Check the messages above and README.md.
goto fail

:fail
popd
pause
exit /b 1

:folder_error
echo Could not open the project folder. Extract the project to a local folder and try again.
pause
exit /b 1
