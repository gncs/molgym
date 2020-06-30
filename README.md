# MolGym

<img src="resources/intro.png" width="40%">

**Reinforcement Learning for Molecular Design Guided by Quantum Mechanics**<br>
Gregor N. C. Simm, Robert Pinsler and José Miguel Hernández-Lobato <br>
*Proceedings of the 37th International Conference on Machine Learning*, Vienna, Austria, PMLR 119, 2020.<br>
https://arxiv.org/abs/2002.07717

The code has been authored by: Gregor N. C. Simm and Robert Pinsler.

## Installation

Dependencies:
* Python  >= 3.6
* PyTorch >= 1.4
* Core dependencies specified in setup.py


1. Create new Python 3.6 environment:
   ```text
   virtualenv --python=python3.6 molgym-venv
   source molgym-venv/bin/activate
   ```

2. Install required packages and library itself:
   ```text
   pip install -r molgym/requirements.txt
   pip install -e molgym/
   ```

3. Install [Sparrow 2.0.1](https://github.com/qcscine/sparrow/releases/tag/2.0.1).

## Usage

The experiments provided in this code base allow learning reinforcement learning agents to design a molecule given a specific bag (single-bag) or multiple bags (multi-bag). See Section 5 of the paper for more details.

1. Single-bag: run the following command 

    ```
    python3 scripts/run.py --name=ch4o --formulas=CH4O --canvas_size=10 --num_steps_per_iter=384 --num_iters=300 --save_rollouts=eval --min_mean_distance=0.95 --max_mean_distance=1.80 --seed=1
    ```
    
    This will automatically generate an experimental directory with the appropriate name, and place the results in the directory. 
    Hyper-parameters for the experiments can be found in the paper.
    
2. Multi-bag: run the following command 

    ```
    python3 scripts/run.py --name=multibag --formulas=H2O,CHN,C2N2,H3N,C2H2,CH2O,C2HNO,N4O,C3HN,CH4,CF4 --canvas_size=10 --num_steps_per_iter=384 --num_iters=500 --save_rollouts=eval --min_mean_distance=0.95 --max_mean_distance=1.80 --seed=1
    ```
    
    This will automatically generate an experimental directory with the appropriate name, and place the results in
    the directory. 
    Hyper-parameters for the experiment can be found in the paper.
    
## Plotting

To generate learning curves, run the command 

```
python3 plot.py
```
    
Running this script will automatically generate a figure of the learning curve in the directory.


## Citation

If you use this code, please cite our [paper](https://arxiv.org/pdf/2002.07717.pdf):
```
@article{Simm2020,
  title={Reinforcement Learning for Molecular Design Guided by Quantum Mechanics},
  author={Simm, Gregor NC and Pinsler, Robert and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:2002.07717},
  year={2020}
}
```
