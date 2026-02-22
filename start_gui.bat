@echo off
title HybridRAG v3 GUI
powershell -NoExit -Command "Invoke-Expression ([System.IO.File]::ReadAllText('%~dp0start_hybridrag.ps1')); python src\gui\launch_gui.py"
