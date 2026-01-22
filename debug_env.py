import gymnasium as gym
from virtual_env import VirtualBattleCityEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    print("--- DEBUGGING ENVIRONMENT ---")
    
    # 1. Direct Instantiation
    print("\n1. Testing Direct Instantiation:")
    try:
        env = VirtualBattleCityEnv(render_mode='rgb_array')
        print(f"Created env: {env}")
        
        obs, info = env.reset() # Gymnasium style reset?
        print(f"Reset return len: 2 (Obs shape: {obs.shape}, Info: {info})")
        
        ret = env.step(0)
        print(f"Step return length: {len(ret)}")
        print(f"Values: {ret}")
        env.close()
    except Exception as e:
        print(f"Direct Instantiation FAILED: {e}")

    # 2. Testing `make_vec_env` (SB3 Wrapper)
    print("\n2. Testing make_vec_env Wrapper:")
    try:
        # Create 1 env
        env = make_vec_env(VirtualBattleCityEnv, n_envs=1, vec_env_cls=DummyVecEnv)
        print(f"Created VecEnv: {env}")
        obs = env.reset()
        print(f"VecEnv Reset obs shape: {obs.shape}")
        
        action = [0]
        ret = env.step(action)
        # VecEnv step usually returns: obs, rewards, dones, infos
        print(f"VecEnv Step return length: {len(ret)}")
        # Check first element (obs)
        print(f"Obs shape: {ret[0].shape}")
        print(f"Rewards: {ret[1]}")
        print(f"Dones: {ret[2]}")
        print(f"Infos len: {len(ret[3])}")
        
        env.close()
        print("VecEnv Test PASSED!")
    except Exception as e:
        print(f"VecEnv Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
