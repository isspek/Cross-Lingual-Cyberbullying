# Cross-Lingual-Cyberbullying
![software image](images/Multi-lang-Hatespeech-Profiling.png)
## Installation
Create a virtual environment:
```bash
virtualenv -p python3.8 venv
```
Activate the environment:
```bash
virtualenv -p python3.8 venv
```
Install the libraries:
```bash
pip install -r requirements.txt
```
The software is ready for the execution

## Training Models
TODO

## Explaining Results
TODO

## PAN Submission

Run the script located in `scripts/pan_run.sh` as follows:

```bash
bash scripts/pan_run.sh {DATASET_FOLDER} {OUTPUT_FOLDER}
```

The script runs the software with a batch size of 2. This could lead slow response. In order to speed up the run, you can increase the batch size.
If you have memory errors, reduce the batch size. 