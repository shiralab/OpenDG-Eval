# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, args):
        super(GroupDRO, self).__init__(args)
        # self.register_buffer("q", torch.Tensor()) # original code
        self.register_buffer("q", torch.Tensor(3))
        self.args = args

    def update(self, minibatches, opt, sch):

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(self.device)

        losses = torch.zeros(len(minibatches)).to(self.device)

        for m in range(len(minibatches)):
            x, y = (
                minibatches[m][0].to(self.device).float(),
                minibatches[m][1].to(self.device).long(),
            )
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.args.groupdro_eta * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {"group": loss.item()}
