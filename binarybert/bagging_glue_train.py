# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function
import argparse
import copy
from bagging_kd_glue import KDLearner
from helper import *
from utils_glue import *
from transformer.tokenization import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from transformer.configuration_bert import BertConfig
from transformer.modeling_dynabert import BertForSequenceClassification
from transformer.modeling_dynabert_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer.modeling_dynabert_binary import BertForSequenceClassification\
    as BertForSequenceClassification_binary
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.binary_model_init import tws_split
# ensemble
import numpy as np
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", default='tmp', type=str, help='jobid to save training logs')
parser.add_argument("--data_dir", default=None, type=str,help="The root dir of glue data")
parser.add_argument("--teacher_model", default='', type=str, help="The teacher model dir.")
parser.add_argument("--student_model", default='', type=str, help="The student model dir.")
parser.add_argument("--task_name", default=None, type=str, help="The name of the glue task to train.")
parser.add_argument("--output_dir", default='output', type=str,help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=None, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

parser.add_argument("--batch_size", default=None, type=int, help="Total batch size for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument('--weight_decay', '--wd', default=0.01, type=float, metavar='W', help='weight decay')
parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")

parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--do_eval", action='store_true')
parser.add_argument('--eval_step', type=int, default=100)

# distillation params
parser.add_argument('--aug_train', action='store_true',
                    help="Whether using data augmentation or not")
parser.add_argument('--kd_type', default='no_kd', choices=['no_kd', 'two_stage', 'logit_kd', 'joint_kd'],
                    help="choose one of the kd type")
parser.add_argument('--distill_logit', action='store_true',
                    help="Whether using distillation over logits or not")
parser.add_argument('--distill_rep_attn', action='store_true',
                    help="Whether using distillation over reps and attns or not")
parser.add_argument('--temperature', type=float, default=1.)

# quantization params
parser.add_argument("--weight_bits", default=32, type=int, help="number of bits for weight")
parser.add_argument("--weight_quant_method", default='twn', type=str,
                    choices=['twn', 'bwn', 'uniform', 'laq'],
                    help="weight quantization methods, can be bwn, twn, laq")
parser.add_argument("--input_bits",  default=32, type=int,
                    help="number of bits for activation")
parser.add_argument("--input_quant_method", default='uniform', type=str, choices=['uniform', 'lsq'],
                    help="weight quantization methods, can be bwn, twn, or symmetric quantization for default")

parser.add_argument('--learnable_scaling', action='store_true', default=True)
parser.add_argument("--ACT2FN", default='gelu', type=str,
                    help='activation fn for ffn-mid. A8 uses uq + gelu; A4 uses lsq + relu.')

# training config
parser.add_argument('--sym_quant_ffn_attn', action='store_true',
                    help='whether use sym quant for attn score and ffn after act') # default asym
parser.add_argument('--sym_quant_qkvo', action='store_true',  default=True,
                    help='whether use asym quant for Q/K/V and others.') # default sym

# hypers clipping threshold
# parser.add_argument('--restore_clip', action='store_true',
#                     help='if true, restore the last step model from rep/attn kd for two stage kd')
parser.add_argument('--clip_init_file', default='threshold_std.pkl', help='files to restore init clip values.')
parser.add_argument('--clip_init_val', default=2.5, type=float, help='init value of clip_vals, default to (-2.5, +2.5).')
parser.add_argument('--clip_lr', default=1e-4, type=float, help='Use a seperate lr for clip_vals / stepsize')
parser.add_argument('--clip_wd', default=0.0, type=float, help='weight decay for clip_vals / stepsize')

# layerwise quantization config
parser.add_argument('--embed_layerwise', default=False, type=lambda x: bool(int(x)))
parser.add_argument('--weight_layerwise', default=True, type=lambda x: bool(int(x)))
parser.add_argument('--input_layerwise', default=True, type=lambda x: bool(int(x)))

### spliting
parser.add_argument('--split', action='store_true',
                    help='whether to conduct tws spliting. NOTE this is only for training binarybert')
parser.add_argument('--is_binarybert', action='store_true',
                    help='whether to use binarybert modelling.')

args = parser.parse_args()
args.do_lower_case = True

log_dir = os.path.join(args.output_dir, 'record_%s.log' % args.job_id)
init_logging(log_dir)

# Prepare devices
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
n_gpu = torch.cuda.device_count()
logging.info("device: {} n_gpu: {}".format(device, n_gpu))

# Prepare seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# Prepare task settings
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
task_name = args.task_name.lower()

# restore the default setting if they are None
if args.batch_size is None:
    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        args.batch_size = int(args.batch_size*n_gpu)
if args.max_seq_length == None:
    if task_name in default_params:
        args.max_seq_length = default_params[task_name]["max_seq_length"]
if task_name not in processors:
    raise ValueError("Task not found: %s" % task_name)
print_args(vars(args))

processor = processors[task_name]()
output_mode = output_modes[task_name]
label_list = processor.get_labels()
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)
config = BertConfig.from_pretrained(args.teacher_model)
config.num_labels = num_labels

student_config = copy.deepcopy(config)
student_config.weight_bits = args.weight_bits
student_config.input_bits = args.input_bits
student_config.weight_quant_method = args.weight_quant_method
student_config.input_quant_method = args.input_quant_method
student_config.clip_init_val = args.clip_init_val
student_config.learnable_scaling = args.learnable_scaling
student_config.sym_quant_qkvo = args.sym_quant_qkvo
student_config.sym_quant_ffn_attn = args.sym_quant_ffn_attn
student_config.embed_layerwise = args.embed_layerwise
student_config.weight_layerwise = args.weight_layerwise
student_config.input_layerwise = args.input_layerwise
student_config.hidden_act = args.ACT2FN

data_dir = os.path.join(args.data_dir,args.task_name)
num_train_optimization_steps = 0
if not args.do_eval:
    if args.aug_train:
        train_examples = processor.get_aug_examples(data_dir)
    else:
        train_examples = processor.get_train_examples(data_dir)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    train_features = convert_examples_to_features(train_examples, label_list,
                                                args.max_seq_length, tokenizer, output_mode)
    train_data, _ = get_tensor_data(output_mode, train_features)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last = True)

    num_train_optimization_steps = int(
        len(train_features) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

eval_examples = processor.get_dev_examples(data_dir)
eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
# eval_sampler = SequentialSampler(eval_data)
# eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

# define the model
print('==> building model',args.task_name,'...')
if task_name == "mnli":
    processor = processors["mnli-mm"]()
    
    mm_eval_examples = processor.get_dev_examples(data_dir)
    mm_eval_features = convert_examples_to_features(
        mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)

    logging.info("***** Running mm evaluation *****")
    logging.info("  Num examples = %d", len(mm_eval_examples))

    mm_eval_sampler = SequentialSampler(mm_eval_data)
    mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                    batch_size=args.batch_size)
else:
    mm_eval_labels = None
    mm_eval_dataloader = None

if not args.do_eval: # need the teacher model for training
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model,config=config)
    teacher_model.to(device)
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
else:
    teacher_model = None

if args.split:
    # rename the checkpoint to restore
    split_model_dir = os.path.join(args.output_dir,'binary_model_init')
    if not os.path.exists(split_model_dir):
        os.mkdir(split_model_dir)
    # copy the json file, avoid over-writing
    source_model_dir = os.path.join(args.student_model, CONFIG_NAME)
    target_model_dir = os.path.join(split_model_dir, CONFIG_NAME)
    os.system('cp -v %s %s' % (source_model_dir, target_model_dir))

    # create the split model ckpt
    source_model_dir = os.path.join(args.student_model, WEIGHTS_NAME)
    target_model_dir = os.path.join(split_model_dir, WEIGHTS_NAME)
    target_model_dir = tws_split(source_model_dir, target_model_dir)
    args.student_model = split_model_dir  # over-write student_model dir
    print("transformed binary model stored at: {}".format(target_model_dir))

# initialize the model
print('==> Load pretrained model from ...')
if args.is_binarybert:
    student_model = BertForSequenceClassification_binary.from_pretrained(args.student_model, WEIGHTS_NAME, config=student_config)
else:
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, WEIGHTS_NAME, config=student_config)
student_model.to(device)
if n_gpu > 1:
    student_model = torch.nn.DataParallel(student_model)

def train(features, save_name="pytorch_model.bin",sample_weights=torch.Tensor(np.ones((10000,1))/10000)):
    """ perform training """
    if args.kd_type == 'joint_kd':
        learner.build()
        learner.train(train_examples, task_name, output_mode, eval_labels,
                        num_labels, train_data, eval_data, eval_examples, tokenizer, save_name, features, sample_weights,
                        mm_eval_dataloader=mm_eval_dataloader, mm_eval_labels=mm_eval_labels)
                        
    elif args.kd_type == 'logit_kd':
        # only perform the logits kd
        learner.build(lr=args.learning_rate)
        learner.args.distill_logit = True
        learner.args.distill_rep_attn = False
        learner.train(train_examples, task_name, output_mode, eval_labels,
                        num_labels, train_data, eval_data, eval_examples, tokenizer, save_name, features, sample_weights,
                        mm_eval_dataloader=mm_eval_dataloader, mm_eval_labels=mm_eval_labels)

    elif args.kd_type == 'two_stage':
        # stage 1: intermediate layer distillation
        learner.args.distill_logit = False
        learner.args.distill_rep_attn = True
        learner.build(lr=2.5*args.learning_rate)
        learner.train(train_examples, task_name, output_mode, eval_labels,
                        num_labels, train_data, eval_data, eval_examples, tokenizer, save_name, features, sample_weights,
                        mm_eval_dataloader=mm_eval_dataloader, mm_eval_labels=mm_eval_labels)

        # stage 2: prediction layer distillation
        learner.student_model.load_state_dict(torch.load(os.path.join(learner.output_dir,save_name+'.bin')))
        learner.args.distill_logit = True
        learner.args.distill_rep_attn = False
        learner.build(lr=args.learning_rate)  # prepare the optimizer again.
        learner.train(train_examples, task_name, output_mode, eval_labels,
                        num_labels, train_data, eval_data, eval_examples, tokenizer, save_name, features, sample_weights,
                        mm_eval_dataloader=mm_eval_dataloader, mm_eval_labels=mm_eval_labels)

    else:
        assert args.kd_type == 'no_kd'
        # NO kd training, vanilla cross entropy with hard label
        learner.build(lr=args.learning_rate)  # prepare the optimizer again.
        learner.train(train_examples, task_name, output_mode, eval_labels,
                        num_labels, train_data, eval_data, eval_examples, tokenizer, save_name, features, sample_weights,
                        mm_eval_dataloader=mm_eval_dataloader, mm_eval_labels=mm_eval_labels)

def sample_models(features, bagging_iters, sample_weights):
    print(str(datetime.datetime.utcnow())+" Start bagging iter: "+str(bagging_iters) )
    print('===> Start retraining ...')
    train(features,str(bagging_iters),sample_weights)

def use_sampled_model(sampled_model, data):
    learner.student_model.load_state_dict(torch.load(sampled_model))
    learner.student_model.eval()
    with torch.no_grad():
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
        student_logits, _, _ = learner.student_model(input_ids, segment_ids, input_mask)

    return student_logits

def combine_softmax_output(pred_test_i, pred_test, alpha_m_mat, i):
    pred_test_delta = alpha_m_mat[0][i] * pred_test_i
    pred_test = torch.add(pred_test, pred_test_delta.cpu())
    return pred_test

def most_common_element(pred_mat):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(args.batch_size):
        counts = np.bincount(pred_mat[i,:])
        pred_most = np.append(pred_most, np.argmax(counts))
    return pred_most

if __name__ == '__main__': # no_kd
    learner = KDLearner(args, device, student_model, teacher_model,num_train_optimization_steps)
    bagging = 2

    if(task_name == 'cola'):
        category = 'mcc'
    elif(task_name == 'sts-b'):
        category = 'corr'
    else:
        category = 'acc'

    # Update sample_weights
    if not args.do_eval:
        index_weak_cls = 0
        features = len(train_examples)
        for i in range(bagging):
            print("bagging "+str(i))
            sample_weights_new = np.random.choice(features, size=features)
            sample_models(features, bagging_iters=i, sample_weights=sample_weights_new)
            print('%s %d-th Sample done !' % (str(datetime.datetime.utcnow()), i))
            index_weak_cls = index_weak_cls + 1
        print("no_kd bagging finished!")

    #use the sampled model
    if task_name == "mnli":
        pred_store = torch.Tensor()
        mm_logits = {0:[],1:[]}
        mm_preds = {0:[],1:[]}
        mm_result = {0:{},1:{}}
        mm_testloader = DataLoader(mm_eval_data, batch_size=args.batch_size, shuffle=False)
        for batch_idx, data in enumerate(mm_testloader):
            data = tuple(t.to(device) for t in data)
            for i in range(2):
                student_model.load_state_dict(torch.load( args.output_dir + "/nokd/" + str(i) + ".bin"))
                student_model.eval()
                with torch.no_grad():
                    input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
                    mm_logits[i], _, _  = student_model(input_ids, segment_ids, input_mask)
                if len(mm_preds[i]) == 0:
                    mm_preds[i].append(mm_logits[i].detach().cpu().numpy())
                else:
                    mm_preds[i][0] = np.append(mm_preds[i][0], mm_logits[i].detach().cpu().numpy(), axis=0)
        pred_store = np.argmax(mm_preds[0][0] + mm_preds[1][0], axis=1)
        final_result = compute_metrics(task_name, pred_store, mm_eval_labels.numpy())
        for i in range(2):
            mm_preds[i] = np.argmax(mm_preds[i][0], axis=1)
            mm_result[i] = compute_metrics(task_name, mm_preds[i], mm_eval_labels.numpy())

        logging.info('Test accuracy from selected model: %f,%f,%f',mm_result[0][category],mm_result[1][category],final_result[category])

    logits = {0:[],1:[]}
    preds = {0:[],1:[]}
    result = {0:{},1:{}}
    pred_store = torch.Tensor()
    testloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)
    for batch_idx, data in enumerate(testloader):
        data = tuple(t.to(device) for t in data)
        for i in range(2):
            student_model.load_state_dict(torch.load( args.output_dir + "/nokd/" + str(i) + ".bin"))
            student_model.eval()
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
                logits[i], _, _ = student_model(input_ids, segment_ids, input_mask)
            if len(preds[i]) == 0:
                preds[i].append(logits[i].detach().cpu().numpy())
            else:
                preds[i][0] = np.append(preds[i][0], logits[i].detach().cpu().numpy(), axis=0)
    if output_mode == "classification":
        pred_store = np.argmax(preds[0][0] + preds[1][0], axis=1)
    elif output_mode == "regression":
        pred_store = np.squeeze((preds[0][0] + preds[1][0])/2)
    final_result = compute_metrics(task_name, pred_store, eval_labels.numpy())
    for i in range(2):
        if output_mode == "classification":
            preds[i] = np.argmax(preds[i][0], axis=1)
        elif output_mode == "regression":
            preds[i] = np.squeeze(preds[i][0])
        result[i] = compute_metrics(task_name, preds[i], eval_labels.numpy())

    logging.info('Test accuracy from selected model: %f,%f,%f',result[0][category],result[1][category],final_result[category])

    # #use the sampled model
    # for num_bagging in range(bagging):
    #     final_result_np = []
    #     final_result = {}
    #     testloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)
    #     for batch_idx, data in enumerate(testloader):
    #         data = tuple(t.to(device) for t in data)
    #         pred_store = torch.Tensor()
    #         for i in range(num_bagging):
    #             sampled_model = args.output_dir + "/nokd/" + str(i) + ".bin"
    #             pred_test_i = use_sampled_model(sampled_model,data)
    #             if(pred_test_i.size()[0]!=(args.batch_size)):
    #                 continue
    #             pred = pred_test_i.max(1, keepdim=True)[1]
    #             pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)
    #         if(pred_test_i.size()[0]!=(args.batch_size)):
    #             continue
    #         pred_most = most_common_element(pred_store.numpy())
    #         print(pred_most)
    #         result = compute_metrics(task_name, pred_most, evaluate_labels[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size].numpy())

    #         final_result_np = np.append(final_result_np, result[category])

    #     print('--------------------'+'num_bagging: '+str(num_bagging)+'--------------------')
    #     print('\n Test accuracy from selected model:',np.mean(final_result_np))

