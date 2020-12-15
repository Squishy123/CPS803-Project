# Fake News Detection CPS803-Project

## [YouTube Video HERE!](https://www.youtube.com/watch?v=CY6PMvMSs0Q)
## [Project Report HERE!](src/summary/report.pdf)

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