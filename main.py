from metaworld.benchmarks import ML1
import time

print(ML1.available_tasks())  # Check out the available tasks

env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`
tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task

obs = env.reset()  # Reset environment

for i in range(1000):
    print('iteration %d'%(i))
    if i%100 ==0:
        obs = env.reset()
    env.render()
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)
    print(obs)  # Step the environoment with the sampled random action
    time.sleep(0.2)
