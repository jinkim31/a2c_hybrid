from hybrid_a2c_agent import HA2Cagent
import gym


def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    agent = HA2Cagent(env)
    agent.train(max_episode_num)
    agent.plot_result()


if __name__ == "__main__":
    main()
