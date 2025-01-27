#!/bin/bash

# Start the scheduler in the background
python -m src.scheduler &

# Start the API server
python -m src.main 