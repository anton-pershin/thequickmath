import numpy as np 

def _assert_even_mesh(x_nodes):
    x_nodes_forward = np.roll(x_nodes, -1)
    dxs = x_nodes_forward - x_nodes
    if not np.isclose(0, np.max(np.abs(1 - dxs[:-1] / dxs[0]))):
        raise Exception('Central differences can only be used on even meshes')

def fd_point(x_nodes, f_nodes, i):
    dx = x_nodes[i + 1] - x_nodes[i]
    dy = f_nodes[i + 1] - f_nodes[i]
    return dy / dx

def bd_point(x_nodes, f_nodes, i):
    dx = x_nodes[i] - x_nodes[i - 1]
    df = f_nodes[i] - f_nodes[i - 1]
    return df / dx

def three_points_midpoint(x_nodes, f_nodes, i):
    _assert_even_mesh(x_nodes[i-1:i+2])
    dx = x_nodes[i] - x_nodes[i - 1]
    return (f_nodes[i+1] - f_nodes[i-1]) / (2*dx)

def three_points_left_endpoint(x_nodes, f_nodes, i):
    _assert_even_mesh(x_nodes[i:i+2])
    dx = x_nodes[i + 1] - x_nodes[i]
    return (-3*f_nodes[i] + 4*f_nodes[i + 1] - f_nodes[i + 2]) / (2*dx)

def three_points_right_endpoint(x_nodes, f_nodes, i):
    _assert_even_mesh(x_nodes[i-2:i+1])
    dx = x_nodes[i] - x_nodes[i - 1]
    return (f_nodes[i - 2] - 4*f_nodes[i - 1] + 3*f_nodes[i]) / (2*dx)

def cd(x_nodes, f_nodes):
    df = np.zeros_like(f_nodes)
    df[0] = three_points_left_endpoint(x_nodes, f_nodes, 0)
    df[1:-1] = np.array([three_points_midpoint(x_nodes, f_nodes, i) for i in range(1, len(f_nodes) - 1)])
    df[-1] = three_points_right_endpoint(x_nodes, f_nodes, len(f_nodes) - 1)
    return df