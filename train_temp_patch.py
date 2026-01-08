class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        # HARD FORCE RENDER
        try:
           # Attempt 1: Standard
           self.training_env.render()
        except:
            pass
            
        try:
           # Attempt 2: Direct Access to Inner NESEnv
           # self.training_env is DummyVecEnv
           # .envs[0] is the BattleCityEnv (wrapped)
           # Let's try to find 'env' recursively
           root = self.training_env.envs[0]
           while hasattr(root, 'env'):
               if hasattr(root, 'render') and 'NESEnv' in str(type(root)):
                   root.render()
                   return True
               root = root.env
           
           # Check unwrapped
           if hasattr(self.training_env.envs[0], 'unwrapped'):
               self.training_env.envs[0].unwrapped.render()
               
        except:
             pass
        return True
