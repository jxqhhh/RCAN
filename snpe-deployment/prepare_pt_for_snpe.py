import model
import argparse
import utility

parser=argparse.ArgumentParser()
args=parser.parse_args()
args.model='RCAN'
args.n_resgroups=10
args.n_resblocks=20
args.n_feats=64
args.pre_train="../RCAN_TestCode/model/RCAN_BIX2.pt"
args.test_only=True
args.load='.'
args.save='RCAN'
args.degradation='BI'
args.reset=False
args.testset='Set5'
args.scale='2'
args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.self_ensemble=False
args.chop=True
args.precision='single'
args.cpu=True
args.n_GPUs=0
args.resume=0
args.save_models=False
args.reduction=16
args.data_train='DIV2K'
args.rgb_range=255
args.n_colors=3
args.res_scale=1
args.print_model=False
checkpoint = utility.checkpoint(args)
m=model.Model(args,checkpoint)

input_shape = [1, 3, 256, 256]
input_data = torch.randn(input_shape)
script_model = torch.jit.trace(m.model, input_data)
mkdir -p dlc/pt
script_model.save("./dlc/pt/RCAN_BIX2.pt")

# optional:
#from DNN_printer import DNN_printer
#DNN_printer(m.model,(3,256,256),1)