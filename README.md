# Reproducing Domain Randomization for Sim-to-Real

Code for domain randomization for sim2real

## Setup

rough

```
sudo apt-get install $(cat apt.txt)
```

```
pip install virtualenv
```

```
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```


For jupyter notebooks
```
pip instatll -r requirements_extra.txt

ipython kernel install --user --name=sim2real

jupyter notebook
```


## Instructions


**Generating data**
```
python3 run_domrand.py

# Kill it when you are done. Killing it can sometimes cause data corruptions for the last file being written, so to be safe, you can run:

./scripts/find_corruption.py
```


**Running model**

```
./run_training.py 
```