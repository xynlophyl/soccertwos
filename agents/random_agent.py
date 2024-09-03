from soccer_twos.environment import SoccerTwosEnv

class randomAgent():

    def __init__(self, env):

        self.env = SoccerTwosEnv(
            root = r'C:/Users/nlinl/Desktop/Important/EECS6892/project/ml-agents/training-envs-executables'
        )

    def train(self):

        # initialize env

        action = self.env.get_random_action()







