import argparse
import time

# from environment import Connect4
from render import Connect4Render
from agent import Agent
from data.dataloader import C4_Dataloader



def run_episode(env, agent1, agent2, a1_label=None, render=False):
    total_steps = 0
    total_a1_rewards, total_a2_rewards = 0, 0

    obs = env.get_obs()

    last_af_a1_obs = obs
    a1_act, a2_act = None, None
    while  True:
        # agent1
        a1_act = agent1.sample(obs)
        af_a1_obs, af_a1_reward, done = env.step(a1_act)
        if done and a1_label and a1_label!="win": af_a1_reward = -af_a1_reward
        # agent1.learn_TDLearning(obs, a1_act, af_a1_reward, af_a1_obs, done)
        agent1.learn_QLearning(obs, a1_act, af_a1_reward, af_a1_obs, done)
        
        total_a1_rewards += af_a1_reward
        if render:
            time.sleep(0.3)
            env.render(a1_act)
        if done:
            # change policy on last step for agent 2
            agent2.learn_TDLearning(last_af_a1_obs, a2_act, -af_a1_reward, obs, done)
            # agent2.learn_QLearning(last_af_a1_obs, a2_act, -af_a1_reward, obs, done)
            total_a2_rewards += -1
            break
        last_af_a1_obs = af_a1_obs

        # agent2
        a2_act = agent2.sample(af_a1_obs)
        af_a2_obs, af_a2_reward, done = env.step(a2_act)
        if done and a1_label and a1_label=="win": af_a2_reward = -af_a2_reward
        # agent2.learn_TDLearning(af_a1_obs, a2_act, af_a2_reward, af_a2_obs, done)
        agent2.learn_QLearning(af_a1_obs, a2_act, af_a2_reward, af_a2_obs, done)
        
        total_a2_rewards += af_a2_reward
        if render:
            time.sleep(0.3)
            env.render(a2_act)
        if done:
            # change policy on last step for agent 1
            agent1.learn_TDLearning(obs, a1_act, -af_a2_reward, af_a1_obs, done)
            # agent1.learn_QLearning(obs, a1_act, -af_a2_reward, af_a1_obs, done)
            total_a1_rewards += -1
            break

        obs = af_a2_obs
        total_steps += 1
    if render:
        env.t.clear()
    return total_steps, total_a1_rewards, total_a2_rewards



def test_episode(env, agent1, agent2, render=False):
    total_steps = 0
    total_a1_rewards, total_a2_rewards = 0, 0

    obs = env.get_obs()
    while  True:

        # agent1
        a1_act = agent1.predict(obs)
        af_a1_obs, af_a1_reward, done = env.step(a1_act)

        total_a1_rewards += af_a1_reward
        if render:
            time.sleep(0.3)
            env.render(a1_act, show_value=True)
        if done:
            total_a2_rewards += -1
            break

        # agent2
        a2_act = agent2.predict(af_a1_obs)
        af_a2_obs, af_a2_reward, done = env.step(a2_act)

        total_a2_rewards += af_a2_reward
        if render:
            time.sleep(0.3)
            env.render(a2_act, show_value=True)
        if done:
            total_a1_rewards += -1
            break

        obs = af_a2_obs
        total_steps += 1
        import ipdb
        ipdb.set_trace()
    if render:
        env.t.clear()
    return total_steps, total_a1_rewards, total_a2_rewards





def main(args, dataloader=None):


    env = Connect4Render()
    agent1 = Agent(player=1, n_state=9175000, n_vec=100, use_mc=False)
    agent2 = Agent(player=2, n_state=9175000, n_vec=100, use_mc=False)

    # render = False
    for eps in range(args.num_episode):
        
        obs, a1_label = None, None
        if dataloader:
            obs, a1_label = dataloader.get_next()

        env.reset(obs)
        ep_steps, ep_a1_rewards, ep_a2_rewards = run_episode(env, agent1, agent2, a1_label, False)

        print("Episode %s: steps = %s, Agent1-reward=%.3f, Agent2-reward=%.3f" %\
              (eps, ep_steps, ep_a1_rewards, ep_a2_rewards))

        # if eps % args.num_render == 0:
        #     render = True
        # else: render = False
        if eps % args.num_test == 0:
            env.reset(obs)
            env.render(reset=True)
            ep_steps, ep_a1_rewards, ep_a2_rewards = test_episode(env, agent1, agent2, True)


    
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episode", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--num_render", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="data/connect-4.data")
    args = parser.parse_args()

    dataloader = None
    if args.data_path:
        dataloader = C4_Dataloader(data_path=args.data_path)

    main(args, dataloader)





