from pydantic import BaseModel


class InvDNConfig(BaseModel):
    model_info_path: str = r'models/inv_dn/weight/model.pdmodel'
    model_weight_path: str = r'models/inv_dn/weight/model.pdiparams'
