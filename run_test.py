from tester import Tester
import dataset
import src.config as config
import os
#os.makedirs("Result", exist_ok=True)

T = Tester(model_path="weights/mobilenetv2/2020/11/06/16:27:35/mobilenetv2.pb")
T.test()