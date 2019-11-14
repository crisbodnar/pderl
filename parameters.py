import parser
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
            self.device = torch.device('cpu')

        # Render episodes
        self.render = cla.render
        self.env_name = cla.env
        self.test = cla.test
        self.save_periodic = cla.save_periodic

        # Number of Frames to Run
        if cla.env == 'Hopper-v2': self.num_frames = 4000000
        elif cla.env == 'Ant-v2': self.num_frames = 6000000
        elif cla.env == 'Walker2d-v2': self.num_frames = 8000000
        elif cla.env == 'HalfCheetah-v2': self.num_frames = 6000000
        elif cla.env == 'Humanoid-v2': self.num_frames = 8000000
        else: self.num_frames = 2000000

        # Synchronization
        if cla.env == 'Hopper-v2' or cla.env == 'Ant-v2' or cla.env == 'Walker2d-v2':
            self.rl_to_ea_synch_period = 1
        else:
            self.rl_to_ea_synch_period = 10
        if cla.sync_period is not None:
            self.rl_to_ea_synch_period = cla.sync_period

        self.opstat = cla.opstat
        self.opstat_freq = cla.opstat_freq

        # Novelty
        self.ns = cla.novelty
        self.ns_epochs = 10

        self.next_save = cla.next_save

        self.ls = 256 if cla.env == 'Humanoid-v2' else 128

        # Imitation learning
        self.imit = cla.imit
        self.imit_agents = cla.imit_agents if cla.imit_agents is not None else 1
        self.imit_buffer_size = cla.imit_buff_size if cla.imit_buff_size is not None else 30000
        self.imit_threshold = cla.imit_thres
        self.distil_type = cla.distil_type
        self.individual_bs = 8000

        # DDPG params
        self.use_ln = True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = cla.seed
        self.batch_size = 128
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        # PER
        self.per = cla.per
        self.buffer_size = 1000000
        self.replace_old = True
        self.alpha = 0.7
        self.beta_zero = 0.5
        self.learn_start = (1 + self.buffer_size / self.batch_size) * 2
        self.total_steps = self.num_frames
        # partition number N, split total size to N part
        self.partition_num = 100

        # ##### NeuroEvolution Params ########
        # Num of trials
        if cla.env == 'Hopper-v2' or cla.env == 'Reacher-v2': self.num_evals = 3
        elif cla.env == 'Walker2d-v2': self.num_evals = 5
        else: self.num_evals = 1

        # Elitism Rate
        if cla.env == 'Reacher-v2' or cla.env == 'Walker2d-v2' or cla.env == 'Ant-v2' or cla.env == 'Hopper-v2':
            self.elite_fraction = 0.2
        else:
            self.elite_fraction = 0.1

        self.pop_size = 10
        assert(self.pop_size > self.imit_agents)
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.mutation_mag = cla.mut_mag
        self.mutation_noise = cla.mut_noise
        self.mutation_batch_size = 256
        self.safe_mut = cla.safe_mut
        self.distil = cla.distil
        self.verbose_mut = cla.verbose_mut
        self.verbose_crossover = cla.verbose_crossover

        # Save Results
        self.state_dim = None; self.action_dim = None  # Simply instantiate them here, will be initialized later
        self.save_foldername = cla.logdir
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

    def write_params(self, stdout=True):
        # Create an empty file in which information about the experiment can be manually written.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.save_foldername, 'info.txt'), 'a') as f:
            f.write(params)