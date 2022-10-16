from icarl import icarl
from FineTune import FineTune
import argparse

def parse_args(): 
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，

    parser.add_argument('--model', type=str, help = "Choose a model: \{MLP/CNN/RNN\}")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--lr', type=float, default=1e-3)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--epochs', type=int, default=20)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--exp_name', type=str, default="exp_default")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--optimizer', type=str, default="Adam")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--w_decay', type=float, default=0.0)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--lamda', type=float, default=1)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--NN', type=bool, default=False)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--st', type=str, default='original')                   # 步骤三，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤四          
    return args

if __name__ == "__main__": 
    args = parse_args()

    if args.model == "FineTune": 
        FineTune(args.exp_name, args.st)
    if args.model == "icarl": 
        icarl(args.lr, args.epochs, args.exp_name, optimizer=args.optimizer, weight_decay=args.w_decay, lamda=args.lamda, NN=args.NN, setting=args.st)