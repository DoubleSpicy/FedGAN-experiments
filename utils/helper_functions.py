from torch.autograd import Variable

def get_torch_variable(arg, cuda=True, cuda_index=0):
    if cuda:
        return Variable(arg).cuda(cuda_index)
    else:
        return Variable(arg)