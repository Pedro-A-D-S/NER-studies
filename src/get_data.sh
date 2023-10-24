#!/bin/bash

# Downloading external data file
wget -P ../data/external https://gmb.let.rug.nl/releases/gmb-2.2.0.zip

# Unzipping data file
unzip ../data/external/gmb-2.2.0.zip

# Move extract data
mv gmb-2.2.0 ../data/raw

# Removing data from external folder
rm ../data/external/gmb-2.2.0.zip