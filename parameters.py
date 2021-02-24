import pprint
import torch
import os


class Parameters:
    def __init__(self, cla, init=True):
        if not init:
            return
        cla = cla.parse_args()

        # Set the device to run on CUDA or CPU
        if not cla.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpfu')

        # Render episodes
        self.render = cla.render
        self.env_name = cla.env
        self.save_periodic = cla.save_periodic

        # Number of Frames to Run
        if cla.env == 'Hopper-v2':
            self.num_frames = 4000000
        elif cla.env == 'Ant-v2' or cla.env == 'Walker2d-v2' or cla.env == 'HalfCheetah-v2':
            self.num_frames = 6000000
        else:
            self.num_frames = 2000000

        # Synchronization
        if cla.env == 'Hopper-v2' or cla.env == 'Ant-v2' or cla.env == 'Walker2d-v2':
            self.rl_to_ea_synch_period = 1
        else:
            self.rl_to_ea_synch_period = 10

        # Overwrite sync from command line if value is passed
        if cla.sync_period is not None:
            self.rl_to_ea_synch_period = cla.sync_period

        # Novelty Search
        self.ns = cla.novelty
        self.ns_epochs = 10

        # Model save frequency if save is active
        self.next_save = cla.next_save

        # DDPG params
        self.use_ln = True
        self.gamma = 0.99
        self.tau = 0.001
        self.seed = cla.seed
        self.batch_size = 128
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.buffer_size = 1000000
        self.ls = 128

        # Prioritised Experience Replay
        self.per = cla.per
        self.replace_old = True
        self.alpha = 0.7
        self.beta_zero = 0.5
        self.learn_start = (1 + self.buffer_size / self.batch_size) * 2
        self.total_steps = self.num_frames

        # ========================================== NeuroEvolution Params =============================================

        # Num of trials
        if cla.env == 'Hopper-v2' or cla.env == 'Reacher-v2':
            self.num_evals = 3
        elif cla.env == 'Walker2d-v2':
            self.num_evals = 5
        else:
            self.num_evals = 1

        # Elitism Rate
        if cla.env == 'Reacher-v2' or cla.env == 'Walker2d-v2' or cla.env == 'Ant-v2' or cla.env == 'Hopper-v2':
            self.elite_fraction = 0.2
        else:
            self.elite_fraction = 0.1

        # Number of actors in the population
        self.pop_size = 10

        # Mutation and crossover
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.mutation_mag = cla.mut_mag
        self.mutation_noise = cla.mut_noise
        self.mutation_batch_size = 256
        self.proximal_mut = cla.proximal_mut
        self.distil = cla.distil
        self.distil_type = cla.distil_type
        self.verbose_mut = cla.verbose_mut
        self.verbose_crossover = cla.verbose_crossover

        # Genetic memory size
        self.individual_bs = 8000

        # Variation operator statistics
        self.opstat = cla.opstat
        self.opstat_freq = cla.opstat_freq
        self.test_operators = cla.test_operators

        # Save Results
        self.state_dim = None  # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.save_foldername = cla.logdir
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=True):
        # Dump all the hyper-parameters in a file.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.save_foldername, 'info.txt'), 'a') as f:
            f.write(params)