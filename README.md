# Installation

In order to run those examples first make sure that you already have installed: 
 - Python 3.8
 - pip 20 or higher
 - virtualenv

First create a virtual environment to run the code:

```bash
virtualenv env_models

```

After that activate the environment and install the dependencies:

```bash
source env_models/bin/activate
pip3 install -r requirements.txt

```

# Usage

Uncoment the line of the code that loads the wanted database and execute the code that you want.

For SARIMA:

```bash
python3 sarima.py

```

For Random Forest:

```bash
python3 random_forest.py

```

# References

The code was based on the examples available by Jason Browlee in his website https://machinelearningmastery.com.
