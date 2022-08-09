
from matplotlib.pyplot import axis
import numpy as np
import matplotlib
import numpy as np
import cv2
import os
dirname = os.path.dirname(__file__)


def visualize(env,Blackbar):



    # Colors
    rgb_colors = {}
    for name, hex in matplotlib.colors.cnames.items():
        rgb_colors[name] = matplotlib.colors.to_rgb(hex)

    
    canvas = np.ones((env.height + 2, env.width + 2, 3)) * 255

    # Fill all but border
    canvas[1:-1,1:-1,:] = 0

    for i , reward in enumerate(env.rewards):
        if env.claimed[i] != 0:
            value, x_, y_ = reward
            if value == 1:
                c = rgb_colors['green']

            else:
                c = rgb_colors['blue']
            i = 0
            while i<3:
                canvas[x_+1,y_+1,i] = c[i]
                i += 1
        
    x, y = env.pos
    i = 0
    while i<3:
        canvas[x+1,y+1,i] = rgb_colors['yellow'][i]
        i += 1


    
    if Blackbar:
        delta = canvas.shape[0] - canvas.shape[1]
        print(delta)
        black_bar = np.zeros((env.height + 2, delta//2))
        print (black_bar.shape)
        a = np.column_stack((black_bar, canvas[:,:,0],black_bar))
        b = np.column_stack((black_bar, canvas[:,:,1],black_bar))
        c = np.column_stack((black_bar, canvas[:,:,2],black_bar))
        canvas = np.stack([a,b,c],axis=2)



    r=cv2.resize(canvas[:,:,0], (84, 84),  interpolation=cv2.INTER_NEAREST)
    g=cv2.resize(canvas[:,:,1], (84, 84),  interpolation=cv2.INTER_NEAREST)
    b=cv2.resize(canvas[:,:,2], (84, 84),  interpolation=cv2.INTER_NEAREST)
    canvas = np.stack([r,g,b],axis=2)
    return canvas
    
if __name__ == '__main__':
    pass



            

