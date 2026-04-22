from yacs.config import CfgNode as CN

_C = CN()

_C.DRUG = CN()
_C.DRUG.EMBEDDING_DIM = 512
_C.DRUG.CONV_DIMS = [1024, 256]
_C.DRUG.MLP_DIMS = [1024, 256]

_C.PROTEIN = CN()
_C.PROTEIN.EMBEDDING_DIM = 512
_C.PROTEIN.INPUT_DIM = 512
_C.PROTEIN.MLP_DIMS = [1024, 256]

_C.MLP = CN()
_C.MLP.INPUT_DIM = 512
_C.MLP.DIMS = [1024, 1024, 512, 2]
_C.MLP.DROPOUT_RATE = 0.2

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.EPOCHS = 100
_C.SOLVER.LR = 5e-5
_C.SOLVER.WEIGHT_DECAY = 0
_C.SOLVER.DROPOUT = 0.3
_C.SOLVER.LOSS_FN = "cross_entropy"
# _C.SOLVER.LOSS_FN = "dirichlet_loss"

_C.DATA = CN()
_C.DATA.TEST_CSV_PATH = "lists/db_new/db_val.csv"
_C.DATA.TRAIN_CSV_PATH = "lists/db_new/db_train.csv"
_C.DATA.VAL_CSV_PATH = "lists/db_new/db_test.csv"
# _C.DATA.TEST_CSV_PATH = "/mnt/data/austin/MocFormer/DrugBank_uni_esm2_3B_pos_len_d200_p700_sim_both_neg_test.csv"
# _C.DATA.TRAIN_CSV_PATH = "/mnt/data/austin/MocFormer/DrugBank_uni_esm2_3B_pos_len_d200_p700_sim_both_neg_train.csv"
# _C.DATA.VAL_CSV_PATH = "/mnt/data/austin/MocFormer/DrugBank_uni_esm2_3B_pos_len_d200_p700_sim_both_neg_val.csv"
_C.DATA.PROTEIN_DIR = "embeddings"
_C.DATA.DRUG_DIR = "drug/embeddings_atomic/"
_C.DATA.PROT_CACHE_SIZE = 4096
_C.DATA.DRUG_CACHE_SIZE = 4096
_C.DATA.NUM_WORKERS = 4
_C.DATA.PREFETCH_FACTOR = 2
_C.DATA.PERSISTENT_WORKERS = True
_C.DATA.PIN_MEMORY = True
_C.DATA.NON_BLOCKING_DEVICE_TRANSFER = True

_C.RESULTS = CN()

def get_cfg_defaults():
    return _C.clone()
