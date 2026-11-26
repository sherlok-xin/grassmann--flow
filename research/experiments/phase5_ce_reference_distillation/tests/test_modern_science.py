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
    def test_saved_teacher_adapter_changes_predictions_and_restores_exactly(self):
        import copy
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from modern_checks import load_adapter_exact
        torch.manual_seed(91)
        cfg = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=1, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8, max_position_embeddings=32)
        base = Qwen3ForCausalLM(cfg).cpu()
        adapter = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                             target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')
        source = get_peft_model(copy.deepcopy(base), adapter).eval()
        target = get_peft_model(copy.deepcopy(base), adapter).eval()
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if 'lora_' in name:
                    parameter.normal_(0., .2)
            tokens = torch.tensor([[1, 2, 3, 4, 5]])
            before = target(tokens).logits.clone()
            wanted = source(tokens).logits.clone()
        weights = {k: v.detach().clone() for k, v in get_peft_model_state_dict(source).items()}
        load_adapter_exact(target, weights)
        with torch.no_grad():
            actual = target(tokens).logits
        self.assertTrue(torch.equal(actual, wanted))
        self.assertFalse(torch.allclose(actual, before))
        incomplete = dict(weights)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(ValueError):
            load_adapter_exact(target, incomplete)

    def test_qwen_lora_and_nested_checkpoint_update_on_toy_cpu_model(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from peft import LoraConfig,get_peft_model
        torch.manual_seed(4)
        cfg=Qwen3Config(vocab_size=41,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=32)
        model=Qwen3ForCausalLM(cfg).cpu()
        model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.config.use_cache=False;model.train()
        params=[p for p in model.parameters() if p.requires_grad]
        before=[p.detach().clone() for p in params]
        opt=torch.optim.AdamW(params,lr=1e-3)
        ids=torch.randint(0,41,(2,12));z=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
        sf=z[:,5:-1].reshape(-1,41);y=ids[:,6:].reshape(-1);teacher=torch.randn_like(sf)
        def fn(s,t,g):return token_loss(s,t,g,mode='kd',lam=1.)[0]
        pieces=[checkpoint(fn,sf[j:j+3],teacher[j:j+3],y[j:j+3],use_reentrant=False).sum()/len(y) for j in range(0,len(y),3)]
        torch.stack(pieces).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in params if p.grad is not None))
        opt.step()
        self.assertTrue(any(not torch.equal(p,b) for p,b in zip(params,before)))
        self.assertTrue(all('lora_' in n for n,p in model.named_parameters() if p.requires_grad))

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
