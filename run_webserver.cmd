@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python pdf2zh\pdf2zh.py -i
pause
