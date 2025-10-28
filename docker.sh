#!/bin/bash
docker build -t pta-sim . 
mkdir data
docker run --rm -v ./data:/project/data pta-sim build

