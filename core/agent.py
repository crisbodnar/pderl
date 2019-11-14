import numpy as np
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
from core import replay_memory
from core import ddpg as ddpg
from scipy.spatial import distance
from scipy.stats import rankdata
from core import replay_memory
from parameters import Parameters
import fastrand
import torch


class Agent:
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Init population
        self.pop = []
        self.buffers = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.GeneticAgent(args))

        # Init RL Agent
        self.rl_agent = ddpg.DDPG(args)
        if args.per:
            self.replay_buffer = replay_memory.PrioritizedReplayMemory(args.buffer_size, args.device,
                                                                       beta_frames=self.args.num_frames)
        else:
            self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size, args.device)

        self.demo_buffer = replay_memory.ReplayMemory(args.imit_buffer_size, args.device)
        self.ounoise = ddpg.OUNoise(args.action_dim)

        self.evolver = utils_ne.SSNE(self.args, self.rl_agent.critic, self.evaluate)

        # Population novelty
        self.ns_r = 1.0
        self.ns_delta = 0.1
        self.best_train_reward = 0.0
        self.time_since_improv = 0
        self.step = 1

        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None

    def evaluate(self, agent: ddpg.GeneticAgent or ddpg.DDPG, is_render=False, is_action_noise=False,
                 store_transition=True, net_index=None):
        total_reward = 0.0
        total_error = 0.0

        state = self.env.reset()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if self.args.render and is_render: self.env.render()
            action = agent.actor.select_action(np.array(state))
            if is_action_noise:
                action += self.ounoise.noise()
                action = np.clip(action, -1.0, 1.0)

            # Simulate one step in environment
            next_state, reward, done, info = self.env.step(action.flatten())
            total_reward += reward

            transition = (state, action, next_state, reward, float(done))
            if store_transition:
                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)

            state = next_state
        if store_transition: self.num_games += 1

        return {'reward': total_reward, 'td_error': total_error}

    def rl_to_evo(self, rl_agent: ddpg.DDPG, evo_net: ddpg.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self):
        epochs = self.args.ns_epochs
        novelties = np.zeros(len(self.pop))
        for _ in range(epochs):
            transitions = self.replay_buffer.sample(self.args.batch_size)
            batch = replay_memory.Transition(*zip(*transitions))

            for i, net in enumerate(self.pop):
                novelties[i] += (net.get_novelty(batch))
        return novelties / epochs

    def train_ddpg(self):
        bcs_loss, pgs_loss = [], []
        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in range(int(self.gen_frames * self.args.frac_frames_train)):
                batch = self.replay_buffer.sample(self.args.batch_size)

                pgl, delta = self.rl_agent.update_parameters(batch)
                pgs_loss.append(pgl)

        return {'bcs_loss': 0, 'pgs_loss': pgs_loss}

    def train(self):
        self.gen_frames = 0
        self.iterations += 1

        # ========================== EVOLUTION  ==========================
        # Evaluate genomes/individuals
        rewards = np.zeros(len(self.pop))
        errors = np.zeros(len(self.pop))
        for i, net in enumerate(self.pop):
            for _ in range(self.args.num_evals):
                episode = self.evaluate(net, is_render=False, is_action_noise=False, net_index=i)
                rewards[i] += episode['reward']
                errors[i] += episode['td_error']

        rewards /= self.args.num_evals
        errors /= self.args.num_evals

        # all_fitness = 0.8 * rankdata(rewards) + 0.2 * rankdata(errors)
        all_fitness = rewards

        # Validation test for NeuroEvolution champion
        best_train_fitness = np.max(rewards)
        champion = self.pop[np.argmax(rewards)]

        # print("Best TD Error:", np.max(errors))

        test_score = 0
        for eval in range(5):
            episode = self.evaluate(champion, is_render=True, is_action_noise=False, store_transition=False)
            test_score += episode['reward']
        test_score /= 5.0

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)

        # ========================== DDPG ===========================
        # Collect experience for training
        self.evaluate(self.rl_agent, is_action_noise=True)

        losses = self.train_ddpg()

        # Validation test for RL agent
        testr = 0
        for eval in range(5):
            ddpg_stats = self.evaluate(self.rl_agent, store_transition=False, is_action_noise=False)
            testr += ddpg_stats['reward']
        testr /= 5

        # Sync RL Agent to NE every few steps
        if self.iterations % self.args.rl_to_ea_synch_period == 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(all_fitness)
            if replace_index == elite_index:
                replace_index = (replace_index + 1) % len(self.pop)

            self.rl_to_evo(self.rl_agent, self.pop[replace_index])
            self.evolver.rl_policy = replace_index
            print('Sync from RL --> Nevo')

        # -------------------------- Collect statistics --------------------------
        return {
            'best_train_fitness': best_train_fitness,
            'test_score': test_score,
            'elite_index': elite_index,
            'ddpg_reward': testr,
            'pg_loss': np.mean(losses['pgs_loss']),
            'bc_loss': np.mean(losses['bcs_loss']),
            'pop_novelty': np.mean(0),
        }


class Archive:
    """A record of past behaviour characterisations (BC) in the population"""

    def __init__(self, args):
        self.args = args
        # Past behaviours
        self.bcs = []

    def add_bc(self, bc):
        if len(self.bcs) + 1 > self.args.archive_size:
            self.bcs = self.bcs[1:]
        self.bcs.append(bc)

    def get_novelty(self, this_bc):
        if self.size() == 0:
            return np.array(this_bc).T @ np.array(this_bc)
        distances = np.ravel(distance.cdist(np.expand_dims(this_bc, axis=0), np.array(self.bcs), metric='sqeuclidean'))
        distances = np.sort(distances)
        return distances[:self.args.ns_k].mean()

    def size(self):
        return len(self.bcs)