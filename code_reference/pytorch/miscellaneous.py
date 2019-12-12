# Average meter
class MovingAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, decay=0.8):
        self.reset()
        self.decay = decay

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.avg * self.decay + val * (1 - self.decay)


# EMA: Including batch norm.
def accumulate(dst, src, bn=False, decay=0.99):
    if hasattr(dst, "module"):
        dst = dst.module

    if hasattr(src, "module"):
        src = src.module

    if bn:
        src_state_dict = src.state_dict()
        dst_state_dict = dst.state_dict()
        params = dict(src.named_parameters()).keys()
        for p in params:
            del src_state_dict[p]

        dst_state_dict.update(src_state_dict)
        dst.load_state_dict(dst_state_dict)

    else:
        dst_params = dict(dst.named_parameters())
        src_params = dict(src.named_parameters())

        for k in dst_params.keys():
            dst_params[k].data.mul_(decay).add_(1-decay, src_params[k].data)
            

# Gradient penalties.            
def compute_zero_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())

    return grad_dout2.view(batch_size, -1).sum(1).mean()


def compute_one_gp(d_out, x_interp):
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_interp,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout = grad_dout.view(grad_dout.size(0), -1)
    slopes = torch.sqrt(torch.sum(torch.pow(grad_dout, 2), dim=1))
    gp = torch.mean(torch.pow(slopes - 1., 2))

    return gp  
