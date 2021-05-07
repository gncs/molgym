# MolGym: Reinforcement Learning for 3D Molecular Design

This repository allows to train reinforcement learning policies for designing molecules directly in Cartesian coordinates. The agent builds molecules by repeatedly taking atoms from a given _bag_ and placing them onto a 3D _canvas_. 

<img src="resources/intro.png" width="40%">

Check out our [blog post](https://mlg-blog.com/2021/04/30/reinforcement-learning-for-3d-molecular-design.html) for a gentle introduction. For more details, see our papers:

**Reinforcement Learning for Molecular Design Guided by Quantum Mechanics**<br>
Gregor N. C. Simm*, Robert Pinsler* and José Miguel Hernández-Lobato<br>
*Proceedings of the 37th International Conference on Machine Learning*, Vienna, Austria, PMLR 108, 2020.<br>
http://proceedings.mlr.press/v119/simm20b.html

**Symmetry-Aware Actor-Critic for 3D Molecular Design**<br>
Gregor N. C. Simm, Robert Pinsler, Gábor Csányi and José Miguel Hernández-Lobato<br>
*International Conference on Learning Representations*, 2021.<br>
https://openreview.net/forum?id=jEYKjPE1xYN

## Setup

Dependencies:
* Python >= 3.7
* torch >= 1.5.1
* [torch-scatter](https://github.com/rusty1s/pytorch_scatter) >= 2.0.5
* [Sparrow 2.0.1](https://github.com/qcscine/sparrow/releases/tag/2.0.1)

Install required packages and library itself:
```
pip install -r requirements.txt
pip install -e .
```

**Note:** Make sure that the CUDA versions associated with `torch` and `torch-scatter` match. Check the [documentation](https://github.com/rusty1s/pytorch_scatter) if you run into any errors when installing `torch-scatter`.

### Sparrow Setup

1. Install the prerequisites. On Ubuntu, run 
```shell
sudo apt-get install cmake libeigen3-dev libboost-all-dev
```
2. Install [Sparrow 2.0.1](https://github.com/qcscine/sparrow/releases/tag/2.0.1):
```shell
git clone git@github.com:qcscine/sparrow.git
cd sparrow
git submodule init
git submodule update
mkdir build install
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSCINE_BUILD_PYTHON_BINDINGS=ON ..
make
make test
make install
export PATH=$PATH:<sparrow directory>/install/bin
```

**Note:** Compiling Sparrow may take a while.

3. Register python bindings:
```shell
export PYTHONPATH=$PYTHONPATH:<sparrow directory>/install/lib/python<version>/site-packages
export SCINE_MODULE_PATH=<sparrow directory>/install/lib
```
For troubleshooting, see the official [documentation](https://scine.ethz.ch/static/download/manuals/v2.0.1/sparrow_manual.pdf).

## Usage

You can use this code to train and evaluate reinforcement learning agents for 3D molecular design. We currently support running experiments given a specific bag (single-bag), a stochastic bag, or multiple bags (multi-bag).

### Training
To perform the single-bag experiment with SF6, run
```shell
python3 scripts/run.py \
    --name=SF6 \
    --symbols=X,F,S \
    --formulas=SF6 \
    --min_mean_distance=1.10 \
    --max_mean_distance=2.10 \
    --bag_scale=5 \
    --beta=-10 \
    --model=covariant \
    --canvas_size=7 \
    --num_envs=10 \
    --num_steps=15000 \
    --num_steps_per_iter=140 \
    --mini_batch_size=140 \
    --save_rollouts=eval \
    --device=cuda \
    --seed=1
```
Hyper-parameters for the other experiments can be found in the papers.

### Evaluation

To generate learning curves, run the following command:
```shell
python3 scripts/plot.py --dir=results
```
Running this script will automatically generate a figure of the learning curve.

To write out the generated structures, run the following command:
```shell
python3 scripts/structures.py --dir=data --symbols=X,F,S
```
You can visualize the structures in the generated XYZ file using, for example, [PyMOL](https://pymol.org/2/).

## Citation

If you use this code, please cite our papers:
```txt
@inproceedings{Simm2020Reinforcement,
  title = {Reinforcement Learning for Molecular Design Guided by Quantum Mechanics},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  author = {Simm, Gregor N. C. and Pinsler, Robert and {Hern{\'a}ndez-Lobato}, Jos{\'e} Miguel},
  editor = {III, Hal Daum{\'e} and Singh, Aarti},
  year = {2020},
  volume = {119},
  pages = {8959--8969},
  publisher = {{PMLR}},
  series = {Proceedings of Machine Learning Research}
  url = {http://proceedings.mlr.press/v119/simm20b.html}
}

@inproceedings{Simm2021SymmetryAware,
  title = {Symmetry-Aware Actor-Critic for 3D Molecular Design},
  author = {Gregor N. C. Simm and Robert Pinsler and G{\'a}bor Cs{\'a}nyi and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
  booktitle = {International Conference on Learning Representations},
  year = {2021},
  url = {https://openreview.net/forum?id=jEYKjPE1xYN}
}
```
