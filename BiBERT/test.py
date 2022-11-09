# import torch
import numpy as np
import time
time_start = time.time()  # 记录开始时间

size = 4259364
pred_numpy = np.random.rand(size)
sample_weights = np.ones((size,1)) / size
target_numpy = np.random.rand(size)
# print('target',target,'\n')
# pred_numpy = np.squeeze(pred_numpy)
miss = [int(x) for x in (pred_numpy != target_numpy)] # not all of GLUE is simple_accuracy
miss2 = [x if x==1 else -1 for x in miss]
miss = np.reshape(np.array(miss),[size,1])
print(miss.shape)
print(sample_weights.shape)
err_m = np.matmul(sample_weights.transpose(),miss) / np.sum(sample_weights)
# err_m = torch.mm(torch.t(sample_weights),miss) / torch.sum(sample_weights)
# print(err_m)

alpha_m = 0.5*np.log((1 - err_m) / float(err_m))

prior_exp = (alpha_m * miss2)
prior_exp = prior_exp.transpose()
sample_weights_new = sample_weights * np.exp(prior_exp)

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)