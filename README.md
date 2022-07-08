# Scrubber: Interactive label debugging for EM data sets

## Data sets

Public EM data sets targeted by this system are available at [Project Magellan data repository](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository).

The private data sets will be made available here as soon as we obtain the required permissions.

## Installation

Please use the file environment.yml under src folder to install a conda virtual environment.

## Usage

1. Create a configuration file to point to your data set. An example can be found under config folder.

2. A jupyter notebook titled [analyst-interface](https://github.com/qcri/scrubber/blob/main/src/analyst-interface.ipynb) is provided for interactive label debugging.

3. Error detectors can be tested using golden labels as well using the [exp_golden.py](https://github.com/qcri/scrubber/blob/main/src/v6/exp_golden.py) program. 
