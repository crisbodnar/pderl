[![Build Status](https://travis-ci.com/crisbodnar/pderl.svg?branch=master)](https://travis-ci.com/crisbodnar/pderl)

# Proximal Distilled Evolutionary Reinforcement Learning

Official code for the AAAI 2020 paper "Proximal Distilled Evolutionary Reinforcement Learning". 

![PDERL](figures/pderl_gif.gif) 

Use the following to cite:

```
@inproceedings{bodnar2020proximal,
  title={Proximal distilled evolutionary reinforcement learning},
  author={Bodnar, Cristian and Day, Ben and Li{\'o}, Pietro},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={3283--3290},
  year={2020}
}
```

#### To Run PDERL #### 

First, you will have to install all the dependencies by running ```pip install -r requirements.txt```.
Additionally, for installing mujoco-py 2.0.2.2, follow the instructions on the [official github](https://github.com/openai/mujoco-py). 

To run PDERL with proximal mutations and distillation-based crossover use:

```bash
python run_pderl.py -env=$ENV_NAME$ -distil -proximal_mut -mut_mag=$MUT_MAG$ -logdir=$LOG_DIR$
```

To evaluate and visualise a trained model in an environment use:

```bash
python play_pderl.py -env=$ENV_NAME$ -model_path=$MODEL_PATH$ -render 
```

#### ENVS TESTED #### 

'Hopper-v2' \
'HalfCheetah-v2' \
'Swimmer-v2' \
'Ant-v2' \
'Walker2d-v2' 

#### CREDITS ####

Our code is largely based on the [code of Khadka and Tumer](https://github.com/ShawK91/erl_paper_nips18) and we would 
like to thank them for making their code publicly available. The proximal mutations code is also relying on 
the [safe mutations code](https://github.com/uber-research/safemutations) of Lehman et al. from Uber Research.