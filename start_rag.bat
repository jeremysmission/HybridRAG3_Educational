@echo off
title HybridRAG v3
powershell -NoExit -Command "Set-ExecutionPolicy -Scope Process Bypass -Force; . '%~dp0start_hybridrag.ps1'"
