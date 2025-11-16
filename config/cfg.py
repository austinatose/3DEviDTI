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
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.EPOCHS = 300
_C.SOLVER.LR = 5e-4
_C.SOLVER.WEIGHT_DECAY = 1e-3
_C.SOLVER.DROPOUT = 0.1
_C.SOLVER.LOSS_FN = "cross_entropy"
# _C.SOLVER.LOSS_FN = "dirichlet_loss"

_C.DATA = CN()
_C.DATA.TEST_CSV_PATH = "lists/overfit_test.csv"
_C.DATA.TRAIN_CSV_PATH = "lists/overfit_train.csv"
_C.DATA.VAL_CSV_PATH = "lists/overfit_val.csv"
_C.DATA.PROTEIN_DIR = "embeddings"
_C.DATA.DRUG_DIR = "drug/embeddings_atomic/"

_C.RESULTS = CN()

def get_cfg_defaults():
    return _C.clone()