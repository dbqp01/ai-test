from transformers import AutoConfig, AutoModelForCausalLM
def load_base(model_id, **kw):
    kw = dict(kw)
    kw.setdefault('trust_remote_code', True)
    kw = dict(kw)
    kw.setdefault('trust_remote_code', True)
    try:
        arch = (AutoConfig.from_pretrained(model_id, trust_remote_code=True).architectures or [''])[0]
        arch = AutoConfig.from_pretrained(model_id).architectures[0]
    except Exception:
        arch = ''
    if ('VL' in arch) or ('Qwen3_5' in arch):
        try:
            from transformers import AutoModelForImageTextToText
            from transformers import AutoModelForImageTextToText as VLM
            VLM = AutoModelForImageTextToText
            return VLM.from_pretrained(model_id, **kw)
            return VLM
        except Exception:
            pass
    return CLM.from_pretrained(model_id, **kw)
    return CLM
CLM = AutoModelForCausalLM
