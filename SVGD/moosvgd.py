import math
import torch
from torch.autograd import Variable
def solve_min_norm_2_loss(grad_1, grad_2):
    v1v1 = torch.sum(grad_1*grad_1, dim=1)
    v2v2 = torch.sum(grad_2*grad_2, dim=1)
    v1v2 = torch.sum(grad_1*grad_2, dim=1)
    gamma = torch.zeros_like(v1v1)
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
    gamma[v1v2>=v1v1] = 0.999
    gamma[v1v2>=v2v2] = 0.001
    gamma = gamma.view(-1, 1)
    g_w = gamma.repeat(1, grad_1.shape[1])*grad_1 + (1.-gamma.repeat(1, grad_2.shape[1]))*grad_2
    # g_w = rays_[:,0].unsqueeze(1) * grad_1 + rays_[:,1].unsqueeze(1)* grad_2

    return g_w

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def kernel_functional_rbf_local(losses, c=1e-2, rays = None, idx =None, binary = True, kernel_type = None):
    n = losses.shape[0]
    n_objs = losses.shape[1]
    ps = []
    for n_obj in range(n_objs):
        ps.append( (losses[:,n_obj, None] - losses[None,:,n_obj]).pow(2))
    new_ray = torch.zeros_like(rays)

# Sử dụng chỉ số từ idx để giữ lại giá trị tương ứng và gán các giá trị khác bằng 0
    if idx is not None:
        for i in range(len(idx)):
            new_ray[i][idx[i]] = 1.0 
    else:
        new_ray = rays

    h_list = []
    for n_obj in range(n_objs):
        if median(ps[n_obj]) != 0.0:
            h_list.append(c*median(ps[n_obj])**2/ math.log(n))
        else: h_list.append(5e-6)

    kernel_matrix = new_ray[:,0]* torch.exp(-ps[0]/h_list[0])
    for i in range(1, n_objs):
        kernel_matrix+= new_ray[:,i]* torch.exp(-ps[i]/h_list[i])

    return kernel_matrix

def kernel_functional_rbf(losses, c=1e-2):
    n = losses.shape[0]
    # print(losses)
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    if median(pairwise_distance) == 0:
        h = c*1e-6
    else:
        h = c*median(pairwise_distance)**2 / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / h) #5e-6 for zdt1,2,3 (no bracket)
    # print(kernel_matrix)
    return kernel_matrix


def get_gradient(grad_1, inputs, losses, alpha, c=1e-2, grad_2 = None, rays = None, idx = None, local_kernel=True):
    # n = inputs.size(0)
    n=losses.shape[0]
    #inputs = inputs.detach().requires_grad_(True)
    
    if grad_2 is not None:
        g_w = solve_min_norm_2_loss(grad_1, grad_2)
    else:
        g_w = grad_1
    # print(g_w.shape)
    ### g_w (100, x_dim)
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    if local_kernel:
        kernel = kernel_functional_rbf_local(losses,c, rays = rays, idx = idx)
    else:
        kernel = kernel_functional_rbf(losses,c)
    # print(kernel)
    grad__ = []
    sum_kernel_col = torch.sum(kernel, dim=1)
    for ker in sum_kernel_col:
        grad_ = []
        ker.backward(retain_graph=True)
        for i,param in enumerate(inputs):
            grad_.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
            param.grad.zero_()
        grad__.append(torch.cat(grad_, dim=0))
    
    grad__ = torch.stack(grad__, dim=0)
    # print(grad__)
    kernel_grad = -alpha*grad__
    gradient = (kernel.to(torch.float32).mm(g_w) - kernel_grad) / n
    # print(gradient)

    return gradient
