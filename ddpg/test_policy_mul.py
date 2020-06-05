import time
import joblib
import os
import os.path as osp
import torch
from utils.logx import EpochLogger


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        #import pdb; pdb.set_trace()
        o, r, d, _ = env.step(a)
        print(r)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            #input() # debug
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv
import numpy as np
class SawyerMultiGoalPickPlaceEnv(SawyerReachPushPickPlaceEnv):
    def __init__(self, random_init=True, **kwargs):
        SawyerReachPushPickPlaceEnv.__init__(
            self,
            random_init=random_init,
            task_type='pick_place')


    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, _, reachDist, _, pushDist, pickRew, _, placingDist = self.compute_reward(action, obs_dict)
        self.curr_path_length +=1

        goal_dist = placingDist if self.task_type == 'pick_place' else pushDist

        if self.task_type == 'reach':
            success = float(reachDist <= 0.05)
        else:
            success = float(goal_dist <= 0.07)

        info = {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': goal_dist, 'success': success}
        info['goal'] = self.goal

        return ob, reward, False, info


    def get_gym_observation_dict(self,obs_dict):
        return dict(
            observation = obs_dict['state_observation'],
            desired_goal = obs_dict['state_desired_goal'],
            achieved_goal = obs_dict['state_achieved_goal']
        )

    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos,self._state_goal))

        return np.concatenate([flat_obs,])



def get_env_func_metaworld(env_name):
    '''Return env function 
    '''
    def env_func():
        env = SawyerMultiGoalPickPlaceEnv() # Create an environment with task `pick_place`
        return env
    
    return env_func



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='../data/ddpg_mul0/ddpg_mul0_s2/')
    parser.add_argument('--len', '-l', type=int, default=150)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    _, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)

    env = get_env_func_metaworld('PickAndPlace-v1')()

    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
