#!/bin/bash
#pyinstaller --onefile   --add-data="venv/lib/python*/site-packages/onnxruntime:onnxruntime" auto-tagger.py
pyinstaller auto-tagger.spec