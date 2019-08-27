#!/bin/bash

#run this in the data folder
find . -name '*.tif' -type f -exec bash -c 'convert "$0" "${0%.tif}.png"' {} \;
find . -name '*.tiff' -type f -exec bash -c 'convert "$0" "${0%.tiff}.png"' {} \;
#find . -name "*.tif" -type f -delete
#find . -name "*.tiff" -type f -delete