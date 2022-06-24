
import numpy as np
import matplotlib
import numpy as np
import os
dirname = os.path.dirname(__file__)


def visualize(env):



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

    return canvas
    
if __name__ == '__main__':
    pass



            

