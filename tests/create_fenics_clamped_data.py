import sys
sys.path.append('../fenics_code')
import clamped_beam_fenics as cbf

# Test plotting. 
coorbefore, coorafter, u = cbf.solve_clamped_beam_fenics()

with open('../data/clamped_beam_fenics.txt', 'w') as file:
    for coor in coorafter:
        file.write(f'{coor[0]}, {coor[1]} \n')


with open('../data/clamped_beam_fenics_before.txt', 'w') as file:
    for coor in coorbefore:
        file.write(f'{coor[0]}, {coor[1]} \n')

with open('../data/clamped_beam_fenics_displacement.txt', 'w') as file:
    for coor in u:
        file.write(f'{coor[0]}, {coor[1]} \n')