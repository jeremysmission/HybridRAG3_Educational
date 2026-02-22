@echo off
title HybridRAG v3 GUI
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\gui\launch_gui.py
pause
