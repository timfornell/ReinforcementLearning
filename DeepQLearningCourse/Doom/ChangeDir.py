import os

DOOM_PATH = "/DeepQLearningCourse/Doom"

def change_dir():
    current_working_dir = os.getcwd()
    if DOOM_PATH not in current_working_dir:
        current_working_dir += DOOM_PATH
        os.chdir(current_working_dir)