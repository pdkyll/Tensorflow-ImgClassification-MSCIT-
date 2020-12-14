from trainer import Trainer
import dataset
import src.config as config
import os
#os.makedirs("Result", exist_ok=True)

T = Trainer()
# if not isinstance(config.pretrained_checkpoint_dir, type(None)):
#     T.restore(config.pretrained_checkpoint_dir)
T.optimize(num_epoch=10)
#T.prune(num_epoch=10)
#T.export()