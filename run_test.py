from tester import Tester
import dataset
import src.config as config
import os
#os.makedirs("Result", exist_ok=True)

T = Tester(model_path="weights/mobilenetv2_tf/2020/11/13/17:25:37/mobilenetv2_tf_pruned_acc=0.6929037053979872.pb")
T.test()
T.estimate_flops()