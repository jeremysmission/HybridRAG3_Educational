@echo off
title HybridRAG v3 GUI
powershell -NoExit -Command "Set-ExecutionPolicy -Scope Process Bypass -Force; . '%~dp0start_hybridrag.ps1'; python src\gui\launch_gui.py"
