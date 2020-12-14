from tester import Tester
import dataset
import src.config as config
import os
#os.makedirs("Result", exist_ok=True)

T = Tester(model_path="weights/mobilenetv2/2020/11/23/17-21-06/mobilenetv2acc=0.8929143772893773.pb")
T.test()
T.estimate_flops()
