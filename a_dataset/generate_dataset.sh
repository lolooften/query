#!/bin/sh
python -B process_db.py
python -B structuralize.py
python -B convert_from_json.py
