"""Identifies which codebase (EDM pixel space or EDM2 SD-VAE latent space) a loaded network
belongs to, from its class name. Checkpoints carry no explicit type field, but
torch_utils.persistence preserves the original class name on unpickle."""

#----------------------------------------------------------------------------
# class name -> codebase ('edm' | 'edm2')

_CODEBASE_BY_CLASS_NAME = {
    # edm/networks.py (EDM, pixel space)
    'EDMPrecond':                    'edm',
    'MidpointCriticPrecond':         'edm',
    # NVIDIA EDM2 checkpoints and edm2/mgm_networks.py (EDM2, SD-VAE latent space)
    'Precond':                       'edm2',
    'EDM_2_Precond_Generator':       'edm2',
    'EDM2_MidpointCriticPrecond':    'edm2',
}

#----------------------------------------------------------------------------

def detect_codebase(net):
    """Return 'edm' or 'edm2' for a loaded network module (or raise if its class is unknown)."""
    name = type(net).__name__
    if name not in _CODEBASE_BY_CLASS_NAME:
        raise ValueError(f'Unknown network class "{name}"; add it to torch_utils.model_registry._CODEBASE_BY_CLASS_NAME')
    return _CODEBASE_BY_CLASS_NAME[name]

#----------------------------------------------------------------------------
