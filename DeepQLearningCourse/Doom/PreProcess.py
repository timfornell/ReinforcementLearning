from skimage import transform# Help us to preprocess the frames
"""
preprocess_frame:
Take a frame.
Resize it.
    __________________
    |                 |
    |                 |
    |                 |
    |                 |
    |_________________|
    
    to
    _____________
    |            |
    |            |
    |            |
    |____________|
Normalize it.

return preprocessed_frame

"""
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10,30:-30]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    
    return preprocessed_frame