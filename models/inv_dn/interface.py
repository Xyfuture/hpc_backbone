from models.inv_dn.model.config import InvDNConfig
from models.inv_dn.model.invdn import InvDN


def load_invdn_model(device:str= 'cpu', config=InvDNConfig()):
    model = InvDN(device,config)
    return model