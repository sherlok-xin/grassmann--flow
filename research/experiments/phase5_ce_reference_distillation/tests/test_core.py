import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from core import project, token_loss, kl_from_logtarget, permute_tail, GoldCache


class ScientificTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.t = torch.randn(7, 13)
        self.y = torch.arange(7)
        self.r = torch.tensor([.1,.8,.2,.9,.01,.7,.95]).log()

    def test_projection_properties(self):
        q, b = project(self.t, self.y, self.r)
        self.assertTrue(torch.allclose(q.exp().sum(-1), torch.ones(7), atol=2e-6))
        self.assertTrue((q.gather(1,self.y[:,None]).squeeze() >= self.r-2e-6).all())
        mask = torch.ones_like(q, dtype=torch.bool).scatter_(1,self.y[:,None],False)
        delta = (q-self.t.log_softmax(-1))[mask].reshape(7,12)
        self.assertTrue(torch.allclose(delta,delta[:,:1].expand_as(delta),atol=2e-6))

    def test_gradient_equivalence(self):
        s = torch.randn_like(self.t, requires_grad=True)
        q,b = project(self.t,self.y,self.r)
        ce = -s.log_softmax(-1).gather(1,self.y[:,None]).squeeze()
        left = ce+5*kl_from_logtarget(s.log_softmax(-1),q)
        right = (1+5*b)*ce+5*(1-b)*kl_from_logtarget(s.log_softmax(-1),self.t.log_softmax(-1))
        gl = torch.autograd.grad(left.mean(),s,retain_graph=True)[0]
        gr = torch.autograd.grad(right.mean(),s)[0]
        self.assertTrue(torch.allclose(gl,gr,atol=2e-6))

    def test_extreme_tail_and_exact_reference_one(self):
        t = torch.tensor([[100.,-100.,-80.],[-100.,100.,-80.]])
        q,b=project(t,torch.tensor([0,0]),torch.tensor([-.0001,0.]))
        self.assertTrue(torch.allclose(q.exp().sum(-1),torch.ones(2)))
        s=torch.randn_like(t,requires_grad=True)
        kl_from_logtarget(s.log_softmax(-1),q).sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_zero_lambda_and_permutation(self):
        s=torch.randn_like(self.t,requires_grad=True)
        a,_=token_loss(s,self.t,self.y,mode='reference',lam=0)
        self.assertTrue(torch.equal(a,torch.nn.functional.cross_entropy(s,self.y,reduction='none')))
        q,_=project(self.t,self.y,self.r)
        qp=permute_tail(q,self.y)
        self.assertTrue(torch.equal(q.sort(-1).values,qp.sort(-1).values))
        self.assertTrue(torch.equal(q.gather(1,self.y[:,None]),qp.gather(1,self.y[:,None])))
        self.assertFalse(torch.equal(q,qp))

    def test_cache_rejects_future_or_misaligned_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            ident={'seed':42,'init_hash':'abc','data_hash':'def','steps':2}
            c=GoldCache(Path(tmp)/'cache',ident,write=True)
            y=self.y[None,:]; r=self.r[None,:]
            c.put(0,[4],y,r)
            read=GoldCache(c.path,ident)
            self.assertTrue(torch.equal(read.get(0,[4],y),r))
            with self.assertRaises(FileNotFoundError): read.get(1,[4],y)
            with self.assertRaises(ValueError): read.get(0,[5],y)
            with self.assertRaises(ValueError): read.get(0,[4],y.flip(1))
            with self.assertRaises(ValueError): GoldCache(c.path,{**ident,'seed':123})

if __name__=='__main__': unittest.main()
