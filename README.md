# Proximal Distilled Evolutionary Reinforcement Learning
Code for the dissertation "Learning-Based Genetic Operators for Reinforcement Learning"

###### Dependencies #######
Python 3.6 \
Pytorch 1.0 \
Numpy 1.15.2 \
Fastrand 1.0 \
Gym 0.12.1 \
Mujoco-py v2.2.0.2 (Requires a license)


#### To Run PDERL #### 
python run_erl.py -env $ENV_NAME$ -distil -safe_mut -mut_mag=0.1 -logidr=$LOG_DIR$

#### ENVS TESTED #### 
'Hopper-v2' \
'HalfCheetah-v2' \
'Swimmer-v2' \
'Ant-v2' \
'Walker2d-v2' 

#### Plots ####

To visualise the plots used in the disseration, run the Jupyter Notebooks from the visualise directory.