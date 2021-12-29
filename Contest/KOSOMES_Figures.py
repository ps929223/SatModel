
##
import matplotlib.pyplot as plt

import Lib.Map as Map

coord_Donghae=Map.sector()['Donghae']
coord_Bound=[coord_Donghae[0]-5, coord_Donghae[1]+5, coord_Donghae[2]-5, coord_Donghae[3]+5]


Map.making_map(coord=coord_Bound,map_res='h',grid_res=5)
plt.tight_layout()