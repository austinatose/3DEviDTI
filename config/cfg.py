from yacs.config import CfgNode as CN

_C = CN()

_C.DRUG = CN()
_C.DRUG.EMBEDDING_DIM = 512
_C.DRUG.CONV_DIMS = [1024, 256]
_C.DRUG.MLP_DIMS = [1024, 256]

_C.PROTEIN = CN()
_C.PROTEIN.EMBEDDING_DIM = 512
_C.PROTEIN.INPUT_DIM = 512
_C.PROTEIN.DIMS = [256]

_C.MLP = CN()
_C.MLP.INPUT_DIM = 512
_C.MLP.DIMS = [1024, 1024, 512, 2]
_C.MLP.DROPOUT_RATE = 0.2


_C.SOLVER = CN()

_C.RESULTS = CN()

def get_cfg_defaults():
    return _C.clone()