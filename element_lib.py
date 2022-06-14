# %%
import numpy as np
import matplotlib.pyplot as plt
import gauss_quad_lib 

# %%
def Element_Linear_Polynomial(x1,y1,x2,y2,x3,y3):
    A = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1] ])
    b1 = np.array([1,0,0])
    [a,b,c] = np.linalg.solve(A,b1)
    return lambda x,y : a*x + b*y +c


# %%
def Element_Linear_Polynomial_coef(x1,y1,x2,y2,x3,y3):
    A = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1] ])
    b1 = np.array([1,0,0])
    [a,b,c] = np.linalg.solve(A,b1)
    return np.array([a,b,c])


# %%
def det(x_val,y_val,x1,y1,x2,y2):
    return (x1-x_val)*(y2-y_val) - (x2-x_val)*(y1-y_val)
def Element_Stiff(x1,y1,x2,y2,x3,y3):
    A = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1] ])
    b1 = np.array([1,0,0])
    b2 = np.array([0,1,0]) 
    grad = np.zeros((2,2))
    [a,b,c] = np.linalg.solve(A,b1)
    grad[0] = [a,b]
    [a,b,c] = np.linalg.solve(A,b2)
    grad[1] = [a,b]
    area_triangle = abs(det(x1,y1,x2,y2,x3,y3))/2
    return np.dot(grad[0],grad[1]) * area_triangle

# %%
def Element_Stiff_self(x1,y1,x2,y2,x3,y3):
    A = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1] ])
    b1 = np.array([1,0,0])
    [a,b,c] = np.linalg.solve(A,b1)
    area_triangle = abs(det(x1,y1,x2,y2,x3,y3))/2
    return (a**2+ b**2) * area_triangle


