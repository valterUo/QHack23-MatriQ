import dimod
from dimod.generators.constraints import combinations
from utils import *


def towards_origo(initial_tensor, dim):
    vartype = dimod.BINARY
    linear = dict()
    quadratic = dict()
    offset = 0.0

    indices_1, indices_2, indices_3 = construct_uvw_tensor(dim)

    # ((x, y), z)
    for i in indices_1:
        coeff = 1
        x = "x" + str(i[0][0])
        y = "y" + str(i[0][1])
        z = "z" + str(i[1])
        pair = str((x, y))
        # Penalize cases when there is difference
        if initial_tensor[i[0][0]][i[0][1]][i[1]] != 0:
            offset += 1
            coeff = -1
            
            # x + 2xy - 4x(x,y) + y - 4y(x,y) + 4(x,y)
            # For each pair x, y we create once the constraint 
            # (2*(x,y) - x - y)^2 = 4(x,y) + x + y - 4(x,y)x - 4(x,y)y + 2xy
            if pair not in linear:
                linear[pair] = 4
                append_linear_safe(x, 1, linear)
                append_linear_safe(y, 1, linear)
                #linear[x] = 1
                #linear[y] = 1

                quadratic[(pair, x)] = -4
                quadratic[(pair, y)] = -4
                quadratic[(x, y)] = 2
        
        append_quadratic_safe((pair, z), coeff, quadratic)

    # ((x, z), y)
    for i in indices_2:
        coeff = 1
        x = "x" + str(i[0][0])
        y = "y" + str(i[1])
        z = "z" + str(i[0][1])
        pair = str((x, z))
        # Penalize cases when there is difference
        if initial_tensor[i[0][0]][i[1]][i[0][1]] != 0:
            offset += 1
            coeff = -1
            
            # (2*(x,z) - x - z)^2 = 4(x,z) + x + z - 4(x,z)x - 4(x,z)z + 2xz
            if pair not in linear:
                linear[pair] = 4
                append_linear_safe(x, 1, linear)
                append_linear_safe(z, 1, linear)
                #linear[x] = 1
                #linear[z] = 1

                quadratic[(pair, x)] = -4
                quadratic[(pair, z)] = -4
                quadratic[(x, z)] = 2
        
        append_quadratic_safe((pair, y), coeff, quadratic)
        

    # ((y, z), x)
    for i in indices_3:
        coeff = 1
        x = "x" + str(i[1])
        y = "y" + str(i[0][0])
        z = "z" + str(i[0][1])
        pair = str((y, z))
        # Penalize cases when there is difference
        if initial_tensor[i[1]][i[0][0]][i[0][1]] != 0:
            offset += 1
            coeff = -1
            
            # (2*(y,z) - y - z)^2 = 4(y,z) + y + z - 4(y,z)y - 4(y,z)z + 2yz
            if pair not in linear:
                linear[pair] = 4
                append_linear_safe(y, 1, linear)
                append_linear_safe(z, 1, linear)
                #linear[y] = 1
                #linear[z] = 1

                quadratic[(pair, y)] = -4
                quadratic[(pair, z)] = -4
                quadratic[(y, z)] = 2
                
        append_quadratic_safe((pair, x), coeff, quadratic)
    
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
    return bqm


def towards_standard(initial_tensor, dim):
    standard = get_standard_tensor_rectangular(dim)
    vartype = dimod.BINARY
    linear = dict()
    quadratic = dict()
    offset = 0.0

    indices_1, indices_2, indices_3 = construct_uvw_tensor(dim)

    # ((x, y), z)
    for i in indices_1:
        coeff = 1
        x = "x" + str(i[0][0])
        y = "y" + str(i[0][1])
        z = "z" + str(i[1])
        pair = (x, y)
        # Penalize cases when there is difference
        if initial_tensor[i[0][0]][i[0][1]][i[1]] != standard[i[0][0]][i[0][1]][i[1]]:
            offset += 1
            coeff = -1
            
            # x + 2xy - 4x(x,y) + y - 4y(x,y) + 4(x,y)
            # For each pair x, y we create once the constraint 
            # (2*(x,y) - x - y)^2 = 4(x,y) + x + y - 4(x,y)x - 4(x,y)y + 2xy
            if (x,y) not in linear:
                linear[(x, y)] = 4
                append_linear_safe(x, 1, linear)
                append_linear_safe(y, 1, linear)
                #linear[x] = 1
                #linear[y] = 1

                quadratic[((x, y), x)] = -4
                quadratic[((x, y), y)] = -4
                quadratic[(x, y)] = 2
        
        append_quadratic_safe((pair, z), coeff, quadratic)

    # ((x, z), y)
    for i in indices_2:
        coeff = 1
        x = "x" + str(i[0][0])
        y = "y" + str(i[1])
        z = "z" + str(i[0][1])
        pair = (x, z)
        # Penalize cases when there is difference
        if initial_tensor[i[0][0]][i[1]][i[0][1]] != standard[i[0][0]][i[1]][i[0][1]]:
            offset += 1
            coeff = -1
            
            # (2*(x,z) - x - z)^2 = 4(x,z) + x + z - 4(x,z)x - 4(x,z)z + 2xz
            if (x, z) not in linear:
                linear[(x, z)] = 4
                append_linear_safe(x, 1, linear)
                append_linear_safe(z, 1, linear)
                #linear[x] = 1
                #linear[z] = 1

                quadratic[((x, z), x)] = -4
                quadratic[((x, z), z)] = -4
                quadratic[(x, z)] = 2
        
        append_quadratic_safe((pair, y), coeff, quadratic)
        

    # ((y, z), x)
    for i in indices_3:
        coeff = 1
        x = "x" + str(i[1])
        y = "y" + str(i[0][0])
        z = "z" + str(i[0][1])
        pair = (y, z)
        # Penalize cases when there is difference
        if initial_tensor[i[1]][i[0][0]][i[0][1]] != standard[i[1]][i[0][0]][i[0][1]]:
            offset += 1
            coeff = -1
            
            # (2*(y,z) - y - z)^2 = 4(y,z) + y + z - 4(y,z)y - 4(y,z)z + 2yz
            if (y, z) not in linear:
                linear[(y, z)] = 4
                append_linear_safe(y, 1, linear)
                append_linear_safe(z, 1, linear)
                #linear[y] = 1
                #linear[z] = 1

                quadratic[((y, z), y)] = -4
                quadratic[((y, z), z)] = -4
                quadratic[(y, z)] = 2
        append_quadratic_safe((pair, x), coeff, quadratic)

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
    return bqm


def towards_user_defined(initial_tensor, target_tensor, dim):
    vartype = dimod.BINARY
    linear = dict()
    quadratic = dict()
    offset = 0.0

    indices_1, indices_2, indices_3 = construct_uvw_tensor(dim)

    # ((x, y), z)
    for i in indices_1:
        coeff = 1
        x = "x" + str(i[0][0])
        y = "y" + str(i[0][1])
        z = "z" + str(i[1])
        pair = (x, y)
        # Penalize cases when there is difference
        if initial_tensor[i[0][0]][i[0][1]][i[1]] != target_tensor[i[0][0]][i[0][1]][i[1]]:
            offset += 1
            coeff = -1
            
            # x + 2xy - 4x(x,y) + y - 4y(x,y) + 4(x,y)
            # For each pair x, y we create once the constraint 
            # (2*(x,y) - x - y)^2 = 4(x,y) + x + y - 4(x,y)x - 4(x,y)y + 2xy
            if (x,y) not in linear:
                linear[(x, y)] = 4
                append_linear_safe(x, 1, linear)
                append_linear_safe(y, 1, linear)
                #linear[x] = 1
                #linear[y] = 1

                quadratic[((x, y), x)] = -4
                quadratic[((x, y), y)] = -4
                quadratic[(x, y)] = 2
        
        append_quadratic_safe((pair, z), coeff, quadratic)

    # ((x, z), y)
    for i in indices_2:
        coeff = 1
        x = "x" + str(i[0][0])
        y = "y" + str(i[1])
        z = "z" + str(i[0][1])
        pair = (x, z)
        # Penalize cases when there is difference
        if initial_tensor[i[0][0]][i[1]][i[0][1]] != target_tensor[i[0][0]][i[1]][i[0][1]]:
            offset += 1
            coeff = -1
            
            # (2*(x,z) - x - z)^2 = 4(x,z) + x + z - 4(x,z)x - 4(x,z)z + 2xz
            if (x, z) not in linear:
                linear[(x, z)] = 4
                append_linear_safe(x, 1, linear)
                append_linear_safe(z, 1, linear)
                #linear[x] = 1
                #linear[z] = 1

                quadratic[((x, z), x)] = -4
                quadratic[((x, z), z)] = -4
                quadratic[(x, z)] = 2
        
        append_quadratic_safe((pair, y), coeff, quadratic)
        

    # ((y, z), x)
    for i in indices_3:
        coeff = 1
        x = "x" + str(i[1])
        y = "y" + str(i[0][0])
        z = "z" + str(i[0][1])
        pair = (y, z)
        # Penalize cases when there is difference
        if initial_tensor[i[1]][i[0][0]][i[0][1]] != target_tensor[i[1]][i[0][0]][i[0][1]]:
            offset += 1
            coeff = -1
            
            # (2*(y,z) - y - z)^2 = 4(y,z) + y + z - 4(y,z)y - 4(y,z)z + 2yz
            if (y, z) not in linear:
                linear[(y, z)] = 4
                append_linear_safe(y, 1, linear)
                append_linear_safe(z, 1, linear)
                #linear[y] = 1
                #linear[z] = 1

                quadratic[((y, z), y)] = -4
                quadratic[((y, z), z)] = -4
                quadratic[(y, z)] = 2
        append_quadratic_safe((pair, x), coeff, quadratic)

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
    return bqm



def construct_point_for_energy_evaluation(initial_tensor, xv, yv, zv, dim):
    v = {}

    for i in range(dim**2):
        for t in ["x", "y", "z"]:
            if t == "x":
                v[t + str(i)] = xv[i]
            if t == "y":
                v[t + str(i)] = yv[i]
            if t == "z":
                v[t + str(i)] = zv[i]

    for x in range(dim**2):
            for y in range(dim**2):
                for z in range(dim**2):
                    if initial_tensor[x][y][z] == 1:
                        v[("x" + str(x), "y" + str(y))] = 1
                    else:
                        if ("x" + str(x), "y" + str(y)) not in v:
                            v[("x" + str(x), "y" + str(y))] = 0

                    if initial_tensor[x][y][z] == 1:
                        v[("y" + str(y), "z" + str(z))] = 1
                    else:
                        if ("y" + str(y), "z" + str(z)) not in v:
                            v[("y" + str(y), "z" + str(z))] = 0

                    if initial_tensor[x][y][z] == 1:
                        v[("x" + str(x), "z" + str(z))] = 1
                    else:
                        if ("x" + str(x), "z" + str(z)) not in v:
                            v[("x" + str(x), "z" + str(z))] = 0
    return v