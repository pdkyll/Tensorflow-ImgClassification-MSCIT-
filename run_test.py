from tester import Tester
import dataset
import src.config as config
import os
#os.makedirs("Result", exist_ok=True)

T = Tester(model_path="weights/lenet5/lenet5.pb")
T.test()