from reaching_env import ReachingEnv                                          
                                                                                
env = ReachingEnv(render_mode='human')
obs, info = env.reset()                                                       
print('Obs initiale:', obs)
                                                                            
for i in range(20000):
    action = env.action_space.sample()  # action aléatoire                    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()                                                              
    if terminated or truncated:
        obs, info = env.reset()                                               
                
env.close()    