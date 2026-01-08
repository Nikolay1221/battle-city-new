from battle_city_env import BattleCityEnv
import numpy as np

def test():
    print("Initializing Env...")
    try:
        env = BattleCityEnv()
    except Exception as e:
        print(f"FAILED to init: {e}")
        return

    print("Resetting...")
    try:
        obs = env.reset()
        print(f"Reset Done. Obs Shape: {obs.shape}, Type: {obs.dtype}")
        print(f"Min: {np.min(obs)}, Max: {np.max(obs)}")
    except Exception as e:
        print(f"FAILED to reset: {e}")
        return

    print("Stepping...")
    try:
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: Reward={reward}, Done={done}, Shape={obs.shape}")
            if done:
                print("Episode Done.")
                env.reset()
    except Exception as e:
        print(f"FAILED to step: {e}")
        return

    env.close()
    print("Test Passed!")

if __name__ == "__main__":
    test()
