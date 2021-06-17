import os
import time
import argparse
import torch
import pandas as pd

from sgm_train import train_document, train_image, train_tabular
from utils import count_run_time


class parameter(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # train parameter
        parser.add_argument('--out_dir', type=str, default='./results/', 
                            help="Output directory.")
        parser.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Initial learning rate.')
        parser.add_argument('--early_stop', action='store_false', default=True,
                            help='Whether to early stop.')
        parser.add_argument('--batch_size', type=int, default=1024,
                            help='Batch size.')
        parser.add_argument('--run_num', type=int, default=10,
                            help='Number of experiments')
        parser.add_argument('--cuda', type=str, default='0',
                            help='Choose cuda')
        parser.add_argument('--seed', type=int, default=42, help="Random seed.")
        
        # train information parameter
        parser.add_argument('--verbose', action='store_false', default=True,
                            help='Whether to print training details')
        parser.add_argument('--print_step', type=int, default=5,
                            help='Epoch steps to print training details')
        parser.add_argument('--plot_logs', action='store_true', default=False,
                            help='Whether to plot training logs')
        # data parameter
        parser.add_argument('--data_name', type=str, default='market',
                            help='Dataset name')
        parser.add_argument('--data_path', type=str, default=f'./data/',
                            help='Wether to inject noise to train data')
        parser.add_argument('--inject_noise', type=bool, default=True,
                            help='Whether to inject noise to train data')
        parser.add_argument('--cont_rate', type=float, default=0.01,
                            help='Inject noise to contamination rate')
        parser.add_argument('--anomal_rate', type=str, default='default',
                            help='Adjust anomaly rate')

        
        # model parameter
        ## General
        parser.add_argument('--lam_out', type=float, default=20,
                            help='Parameter Lambda_outliers')
        parser.add_argument('--lam_dist', type=float, default=0.01,
                            help='Parameter Lambda_DE')
        parser.add_argument('--a', type=float, default=15,
                            help='Parameter a')
        parser.add_argument('--epsilon', type=float, default=90,
                            help='Parameter epsilon')
        # Specific
        parser.add_argument('--model_name', type=str, default='SGM',
                            help='Choose model')
        parser.add_argument('--hidden_dim', type=str, default='auto',
                            help='Hidden dimension of the model')
        
        if __name__ == '__main__':
            args = parser.parse_args()
        else:
            args = parser.parse_args([])
            
        args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            
        # Specific design        
        
        self.__dict__.update(args.__dict__)
        
    def update(self, update_dict):
        logs = '==== Parameter Update \n'
        origin_dict = self.__dict__
        for key in update_dict.keys():
            if key in origin_dict:
                logs += f'{key} ({origin_dict[key]} -> {update_dict[key]}), '
                origin_dict[key] = update_dict[key]
            else:
                logs += f'{key} ({update_dict[key]}), '
        self.__dict__ = origin_dict
        print(logs)


if __name__ == '__main__':
    
    start_time = time.time()
    # Total metrics
    metrics = pd.DataFrame()
    
    
    # Conduct one experiements
    args = parameter()
    print(f'Device is {args.device.type}-{args.cuda}')
    an_metrics_dict = train_tabular(args)
    metrics = pd.DataFrame(an_metrics_dict, index=[0])


    # Conduct multiple experiments
    # One parameters
#     lam_out = [1, 2, 3]
    
#     for param in lam_out:
#         # init
#         args = parameter()
#         # update
#         update_dict = {'lam_out': param}
#         args.update(update_dict)
#         an_metrics_dict = train_image(args)
#         an_metrics = pd.DataFrame(an_metrics_dict, index=[f'{param}'])
#         metrics = pd.concat([metrics, an_metrics])
    
    
    
    # Two parameters
    # Iterate parameters
    
#     lam_out = [1, 2, 3, 4, 5]
#     lam_dist = [0.06, 0.04, 0.02, 1e-2, 8e-3, 1e-3]
    
#     print(f'lam_out: {lam_out}')
#     print(f'lam_dist: {lam_dist}')
    
#     args = parameter()
    
#     count_time = count_run_time(5 * 6)
#     count_time.path = f'{args.out_dir}{args.model_name}_{args.data_name}_{time.strftime("%M%S")}.txt'
    
#     for param1 in lam_out:
#         for param2 in lam_dist:
#             # init
#             args_tmp = parameter()
#             args_tmp.update(args.__dict__)
            
#             # update
#             update_dict = {'lam_out': param1, 'lam_dist': param2}
#             args_tmp.update(update_dict)
#             print(f'Train Parameter: {args_tmp.__dict__}')
#             an_metrics_dict = train_tabular(args_tmp)
#             an_metrics = pd.DataFrame(an_metrics_dict, index=[f'{param1}_{param2}'])
#             metrics = pd.concat([metrics, an_metrics])
            
#             count_time.current_count()
            
            
    # Three parameters
    
#     epsilon = [80, 84, 86, 88, 90]
#     lam_out = [3, 4, 5, 6, 7, 10, 18]
#     lam_dist = [0.1, 0.05, 0.02, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    
#     args = parameter()
#     count_time = count_run_time(5 * 7 * 8)
#     count_time.path = f'{args.out_dir}{args.model_name}_{args.data_name}_{str(time.time()).split(".")[0][-2:]}.txt'

#     for param0 in epsilon:
#         for param1 in lam_out:
#             for param2 in lam_dist:
                                
#                 # init
#                 args_tmp = parameter()
#                 args_tmp.update(args.__dict__)
                
#                 # update
#                 update_dict = {'epsilon': param0, 'lam_out': param1, 'lam_dist': param2}
#                 args_tmp.update(update_dict)
                
#                 # train
#                 print(f'Train Parameter: {args_tmp.__dict__}')
#                 an_metrics_dict = train_tabular(args_tmp)
#                 an_metrics = pd.DataFrame(an_metrics_dict, index=[f'{param0}_{param1}_{param2}'])
#                 metrics = pd.concat([metrics, an_metrics])
                
#                 count_time.current_count()
    
    
    
    
    
    
    
    
    
    
    
# #     anomal_rate = [0.05, 0.1, 0.15, 0.20, 0.25]
# #     lam_out = [2, 4, 5]
# #     lam_dist = [8e-5, 5e-5, 1e-5, 5e-6]
    
# #     count_time = count_run_time(5 * 3 * 4)

# #     for param0 in anomal_rate:
        
# #         for param1 in lam_out:
# #             for param2 in lam_dist:
                
# #                 if param0 <= 0.1:
# #                     epsilon = 90
# #                 elif param0 <= 0.2:
# #                     epsilon = 80
# #                 elif param0 <= 0.3:
# #                     epsilon = 70
                
# #                 # init
# #                 args = parameter()
# #                 count_time.path = f'{args.out_dir}{args.model_name}_{args.data_name}_{time.strftime("%M%S")}.txt'
                
# #                 # update
# #                 update_dict = {'lam_out': param1, 'lam_dist': param2, 'anomal_rate': param0}
# #                 args.update(update_dict)
                
# #                 # specific
# #                 update_dict = {'batch_size': 32, 'epsilon': epsilon, 'hidden_dim': '[1024, 256, 80, 20]'} # For document
# #                 args.update(update_dict)
                
# #                 # train
# #                 print(f'Train Parameter: {args.__dict__}')
# #                 an_metrics_dict = train_document(args)
# #                 an_metrics = pd.DataFrame(an_metrics_dict, index=[f'{param0}_{param1}_{param2}'])
# #                 metrics = pd.concat([metrics, an_metrics])
                
# #                 count_time.current_count()
    

    print(f'Finished!\nTotal time is {time.time()-start_time:.2f}s')
    print(f'Current time is {time.strftime("%m%d_%H%M")}')
    print(metrics.sort_values('AUC', ascending=False))
    metrics.to_csv(f'{args.out_dir}{args.model_name}_{args.data_name}_{time.strftime("%m%d_%H%M")}.csv')