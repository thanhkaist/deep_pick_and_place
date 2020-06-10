DPDG for pick and place robot task
===================

### Requirement

- metaworld
- mujoco license
- tqdm

Check requirement.txt for packages version.

### Files

- ddpg.py Train script for DDPG with pick and place (fixed object pos and goal pos)
- ddpg_multi.py Train script for DDPG with multitask pick and place (random object pos and random goal pos)
- test_policy.py Test script for DDPG with pick and place
- test_policy_mul.py Test script for DDP with multitask pick and place 

### TRAIN

```python ddpg.py --env <env_name> --exp_name <exp name> --seed <seed>```


```python ddpg_mul.py --env <env_name> --exp_name <exp name> --seed <seed>```

### TEST

```python test_policy.py --fpath <path to log folder e.g ../data/ddpg/ddpg_s0> --len <max ep len> ```

```python test_policy_mul.py --fpath <path to log folder e.g ../data/ddpg/ddpg_s0> --len <max ep len> ```

### Reference

DDPG+HER - pytorch

https://github.com/TianhongDai/hindsight-experience-replay

DDPG+HER - tensorflow

https://github.com/jangirrishabh/Overcoming-exploration-from-demos
