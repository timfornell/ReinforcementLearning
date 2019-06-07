import os
import tensorflow as tf
from StackFrames import *
from CreateEnv import *
from Parameters import *
from DeepQNetwork import *

os.chdir("/Users/timfornell/Documents/GitHub/ReinforcementLearning/DeepQLearningCourse/Doom")

DQNetwork = DQNetwork(state_size, action_size, learning_rate)
# Reset the graph
tf.reset_default_graph()

writer = tf.summary.FileWriter("log/")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    game, possible_actions = create_environment()
    
    totalScore = 0
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(1):
        
        done = False
        
        game.new_episode()
        
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            
            if done:
                break  
                
            else:
                print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
                
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()