from trainer import Trainer
import dataset
import src.config as config
import os
#os.makedirs("Result", exist_ok=True)

T = Trainer()
T.optimize(num_epoch=30)
T.export()