import torch

import utility
import data
import model
import loss
from option import args
import os
import math
from decimal import Decimal
from math import ceil

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
import imageio

CPU_latency = 154.8
GPU_latency = 226.8
DSP_latency = 32.4

class TVAnalyzer():
    def __init__(self, args, loader, my_model, my_weak_model, ckp):
        self.args = args
        self.scale = args.scale

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.weak_model = my_weak_model

        self.log_folder = "../experiment/TV_threshold_{}".format(self.args.threshold)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

    def test(self):
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_latency = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):

                    time_cpu = time_gpu = time_dsp = 0

                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    sr = torch.zeros(hr.shape)
                    if not no_eval:
                        lr, hr, sr = self.prepare([lr, hr, sr])
                    else:
                        lr, sr = self.prepare([lr, sr])
                    

                    # Divide lr into patches
                    row_num = ceil((lr.shape[2]-self.args.overlapping_size)/(self.args.patch_height-self.args.overlapping_size))
                    col_num = ceil((lr.shape[3]-self.args.overlapping_size)/(self.args.patch_width-self.args.overlapping_size))

                    for rowIdx in range(row_num):
                        for colIdx in range(col_num):
                            row_start = rowIdx*(self.args.patch_height-self.args.overlapping_size)
                            row_end = min(rowIdx*(self.args.patch_height-self.args.overlapping_size)+self.args.patch_height, lr.shape[2])
                            col_start = colIdx*(self.args.patch_width-self.args.overlapping_size)
                            col_end = min(colIdx*(self.args.patch_width-self.args.overlapping_size)+self.args.patch_width, lr.shape[3])
                            lr_patch = lr[:, :, row_start:row_end, col_start:col_end]
                            
                            def calcTV(patch):
                                assert patch.shape[0] == 1
                                TV = 0
                                for c in range(patch.shape[1]):
                                    for i in range(patch.shape[2]-1):
                                        for j in range(patch.shape[3]-1):
                                            TV += abs(patch[0][c][i+1][j]-patch[0][c][i][j])
                                            TV += abs(patch[0][c][i][j+1]-patch[0][c][i][j])
                                TV /= (patch.nelement())
                                return TV

                            TV = calcTV(lr_patch)
                            print("TV={} threshold={}".format(TV, self.args.threshold))
                            if TV < self.args.threshold:
                                sr_patch = self.model(lr_patch, idx_scale)
                                if time_cpu + CPU_latency < time_gpu + GPU_latency:
                                    time_cpu += CPU_latency
                                else:
                                    time_gpu += GPU_latency
                            else:
                                sr_patch = self.weak_model(lr_patch, idx_scale)
                                time_dsp += DSP_latency

                            sr[:, :, scale*row_start:scale*row_end, scale*col_start:scale*col_end] += sr_patch

                    

                    eval_latency += max(time_cpu, time_gpu, time_dsp)
                    
                    for rowIdx in range(row_num-1):
                        row_start = scale*(rowIdx+1)*(self.args.patch_height-self.args.overlapping_size)
                        row_end = row_start+scale*self.args.overlapping_size
                        sr[:, :, row_start:row_end, :] /= 2
                    for colIdx in range(col_num-1):
                        col_start = scale*(colIdx+1)*(self.args.patch_width-self.args.overlapping_size)
                        col_end = col_start+scale*self.args.overlapping_size
                        sr[:, :, :, col_start:col_end] /= 2
                    
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])
                    
                    imageio.imsave('{}.png'.format(idx_img), sr[0].byte().permute(1, 2, 0).cpu().numpy())

                    #if self.args.save_results:
                        #self.ckp.save_results_nopostfix(filename, save_list, scale) #TODO

                #self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test) #TODO
                #best = self.ckp.log.max(0) #TODO
                print( 
                    '[{} x{}]\tPSNR: {:.3f}\tLatency:{}'.format(
                        self.args.data_test,
                        scale,
                        eval_acc / len(self.loader_test),
                        eval_latency / len(self.loader_test)
                    )
                ) #TODO

        #self.ckp.write_log(
        #    'Total time: {:.2f}s, ave time: {:.2f}s\n'.format(timer_test.toc(), timer_test.toc()/len(self.loader_test)), refresh=True
        #) #TODO
        #if not self.args.test_only:
        #    self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch)) #TODO

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

if __name__ == '__main__':
    torch.manual_seed(args.seed)

    loader = data.Data(args)
    checkpoint = utility.checkpoint(args)
    strong_model = model.Model(args, checkpoint)

    args.conv = args.weak_model_conv
    args.pre_train = args.weak_model_pre_train
    weak_model_checkpoint = utility.checkpoint(args)
    weak_model = model.Model(args, weak_model_checkpoint)

    t = TVAnalyzer(args, loader, strong_model, weak_model, checkpoint)
    t.test()

