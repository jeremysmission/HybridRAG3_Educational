@echo off
title HybridRAG v3
powershell -NoExit -Command "Invoke-Expression ([System.IO.File]::ReadAllText('%~dp0start_hybridrag.ps1'))"
