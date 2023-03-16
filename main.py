from runner import train
from maddpg import MADDPG
from helper import make_env, get_args, set_seed


# if __name__ == '__main__':

set_seed(78)
args = get_args()
env, dim_info = make_env(args)

train(args, env, dim_info)