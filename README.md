# CPS803-Project
Fake News Detection

Checkout [Contributions](CONTRIBUTIONS.md)
# Running Code
First activate the environment
``` 
conda activate fakenews
```

Next navigate to the model you want to run and call the python file
```
cd src/models/svm
python svm_clement_bigram.py
```
There is also a bootstrapped batch script that can be executed.

# First Creating Environment
```
conda env create -f environment.yml
```

# Activating and Deactivation Conda Environment
```
conda activate fakenews
```
```
conda deactivate fakenews
```

# Adding a package and updating environment.yml
```
conda install pkg_name
```

``` 
conda env export --from-history > environment.yml
```