"""Response masking and checkpointed exact-KL tests, independent of large models."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'scripts'))
from core import token_loss


class ModernScience(unittest.TestCase):
    def test_checkpointed_response_chunks_match_dense_gradients(self):
        torch.manual_seed(7)
        z=torch.randn(2,13,31,requires_grad=True);t=torch.randn_like(z);labels=torch.randint(0,31,(2,13))
        labels[:,:6]=-100;labels[1,10:]=-100
        mask=labels[:,1:]!=-100;gold=labels[:,1:][mask]
        reference=torch.full((len(gold),),-.8)
        sf=z[:,:-1][mask];tf=t[:,:-1][mask]
        dense,_=token_loss(sf,tf,gold,mode='reference',lam=1.,reference_logp=reference)
        gd=torch.autograd.grad(dense.mean(),z)[0]
        zz=z.detach().clone().requires_grad_();ss=zz[:,:-1][mask];pieces=[]
        def fn(s,t,y,r):return token_loss(s,t,y,mode='reference',lam=1.,reference_logp=r)[0]
        for j in range(0,len(gold),3):
            sl=slice(j,j+3)
            pieces.append(checkpoint(fn,ss[sl],tf[sl],gold[sl],reference[sl],use_reentrant=False).sum()/len(gold))
        gc=torch.autograd.grad(torch.stack(pieces).sum(),zz)[0]
        self.assertTrue(torch.allclose(gd,gc,atol=1e-7,rtol=1e-5))
        self.assertEqual(float(gc[:,:5].abs().sum()),0.)

    def test_frozen_split_masks_and_document_disjointness(self):
        path=HERE/'modern_data.json'
        if not path.exists():self.skipTest('data not prepared yet')
        m=json.loads(path.read_text());seen=set()
        for split,e in m['files'].items():
            # Data integrity only, never predictions or test-based configuration choice.
            with np.load(ROOT/e['path']) as d:
                x,y,att=d['input_ids'],d['labels'],d['attention_mask']
                self.assertTrue(np.array_equal(y[y!=-100],x[y!=-100]))
                self.assertTrue(np.all(y[att==0]==-100))
                self.assertEqual(int((y!=-100).sum()),e['unique_response_targets'])
                for i,line in enumerate((ROOT/e['records_path']).read_text().splitlines()):
                    r=json.loads(line);n=int(att[i].sum());k=r['prompt_length']
                    self.assertTrue(np.all(y[i,:k]==-100));self.assertTrue(np.all(y[i,k:n]!=-100))
                    self.assertEqual(int(y[i,n-1]),m['response_end_token_id'])
                    self.assertNotIn(r['group_hash'],seen)
                groups={json.loads(line)['group_hash'] for line in (ROOT/e['records_path']).read_text().splitlines()}
                seen.update(groups)


if __name__=='__main__':unittest.main()
