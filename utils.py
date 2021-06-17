import time

class count_run_time(object):
    def __init__(self, all_num):
        self.cur_num = 1
        self.start_time = time.time()
        self.all_num = all_num
        self.path = './time_count.txt'
        
    def current_count(self):
        '''Usage:
            count_time = count_run_time(5 * 4 * 4)
            count_time.path = f'{args.out_dir}{args.model_name}_{args.data_name}.txt'
            main()
            count_time.current_count()
        '''

        past_time = time.time()-self.start_time
        avg_time = past_time / self.cur_num
        fut_time = avg_time * (self.all_num - self.cur_num)

        content = f'Current Num: {self.cur_num} / {self.all_num}\n'
        content += f'Past time: {past_time:.2f}s ({past_time/3600:.2f}h)\n'
        content += f'Average time: {avg_time:.2f}s ({avg_time/3600:.2f}h)\n'
        content += f'Future time: {fut_time:.2f}s ({fut_time/3600:.2f}h)\n'

        with open(self.path, 'w') as f:
            f.write(content) 
            
        self.cur_num += 1
        
import os
import re
def check_run_pid():
    gpu_status = os.popen('nvidia-smi').read()
    pid = re.findall(' \d{2,5} ', gpu_status)
    print('==== Current run pid:')
    idx = 0
    for i in pid:
        try:
            content = [x for x in os.popen(f'ps -aux|grep "\ {i.strip()}\ "').read().split('\n') if 'main' in x]
            if content:
                print(f'[{idx}]', content[0])
                idx += 1
        except:
            print(os.popen(f'ps -aux|grep {i}').read().split('\n'))
        

def check_run_time():
    all_file = next(os.walk('./results/'))[2]
    txt_file = [i for i in all_file if i[-3:] == 'txt']
    print('==== Current run time:')
    for idx, txt in enumerate(txt_file):
        with open(f'./results/{txt}', 'r') as f:
            content = f.read().split('\n')
            filename = txt[:-4].split('_')
            print(f'[{idx}] {filename[0]:8s}: {filename[1]:>10s}, {content[0].split(":")[1]:>10s}', end="  ")
            print(f'--Past {content[1].split(":")[1]:>20s},   --Average {content[2].split(":")[1]:>20s}', end="  ")   
            print(f'--Future {content[3].split(":")[1]:>20s}') 