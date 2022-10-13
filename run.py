def parse_args(): 
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，

    parser.add_argument('--model', type=str, help = "Choose a model: \{MLP/CNN/RNN\}")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--lr', type=float, default=1e-3)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--train_epochs', type=int, default=10)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--alpha', type=float, default=1.0)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--batch_size', type=int, default=256)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--save_dir', type=str, default="")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--layer_num', type=int, default=1)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--hidden_size', type=int, default=128)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--data_path', type=str, default="MNIST_data/")                   # 步骤三，后面的help是我的描述
    parser.add_argument('--activation', type=str, default="R")                   # 步骤三，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤四          
    return args

def parse_yaml(model): 
    f = open("config/default_" + str(model) + ".yml", "r" ) 
    y = yaml.load(f, Loader=yaml.FullLoader) 
    return y 

if __name__ == "__main__": 