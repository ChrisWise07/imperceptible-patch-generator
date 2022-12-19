# Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies
![](robust_dpatch_imperceptible_patch_compare.png)
### How to use:
#### Clone the repository:
```bash
git clone https://github.com/ChrisWise07/imperceptible-patch-generator.git
```
#### Move into the root directory:
```bash
cd <path-to-repo>/imperceptible-patch-generator
``` 
#### Create a virtual environment (optional):
```bash
python3 -m venv ./venv
```
#### Activate the virtual environment (optional):
```bash
source venv/bin/activate
```
#### Install required files:
```bash
pip install -r requirements.txt
```
#### Run:
```bash
bash ./execute_imperceptible_patch.sh
```
#### View Results:
```bash
cd ./code_and_experiment_data/experiment_data/<experiment-name>
```
#### View the hyper-parameters:
To see the list of hyper-parameters and default values, run:
```bash
python ./code_and_experiment_data/main.py --help
``` 
#### Modify the hyper-parameters:
To modify the hyper-parameters, open the file `<path-to-repo>/imperceptible-patch-generator/execute_imperceptible_patch.sh` and modify by adding the following to the file:
```bash
--<hyper-parameter-name> <hyper-parameter-value>
```
### Credit
The code has adapted two libraries:
<br />https://github.com/Trusted-AI/adversarial-robustness-toolbox  
https://github.com/ZhengyuZhao/PerC-Adversarial  
