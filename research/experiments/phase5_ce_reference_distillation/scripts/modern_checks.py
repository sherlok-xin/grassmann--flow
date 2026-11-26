"""Checks that an adapted teacher/S0 really receives its saved adapter tensors."""


def load_adapter_exact(model, weights):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected):
        raise ValueError('adapter checkpoint keys do not match the configured trainable adapter')
    if any(weights[key].shape != expected[key].shape for key in expected):
        raise ValueError('adapter checkpoint tensor shapes do not match')
    result = set_peft_model_state_dict(model, weights)
    actual = get_peft_model_state_dict(model)
    for key, tensor in actual.items():
        wanted = weights[key].detach().to(device='cpu', dtype=tensor.dtype)
        if not torch.equal(tensor.detach().cpu(), wanted):
            raise AssertionError('saved adapter tensor was not restored exactly: ' + key)
    return result
