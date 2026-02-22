@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo Installing pyinstaller...
pip install pyinstaller

echo Starting PyInstaller build...
REM --noconfirm: Overwrite existing build
REM --onedir: Create a one-folder bundle (default behavior, but explicitly stated)
REM --console: Open a console window when running
pyinstaller --noconfirm --onedir --console --name pdf2zh --hidden-import=pdf2zh --collect-all gradio --collect-all gradio_client --collect-all safehttpx --collect-all groovy --collect-all gradio_pdf --collect-all babeldoc --distpath dist pdf2zh\pdf2zh.py

echo.
echo =========================================================================
echo Build complete! 
echo The compiled "pdf2zh" application is located in the "dist\pdf2zh" folder.
echo You can now copy the entire "dist\pdf2zh" folder to another Windows PC 
echo and run pdf2zh.exe directly without needing Python installed.
echo =========================================================================
pause
