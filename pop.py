class mp():
    '''
    multi-processing interface
    '''
    def __init__(self, task_func, param_list, n_process=4, using_tqdm=False, tqdm_desc='mp2'):
        self.task_func = task_func
        self.param_list = param_list
        self.n_process = n_process
        self.using_tqdm = using_tqdm
        self.tqdm_desc = tqdm_desc
        
    def run(self):
        self.result = []
        with Pool(self.n_process) as pool, Manager() as manager:
            # communication channel between process / with parent
            # predefined codes
            # 1. 'break' : True - stop all children and return 
            comm = manager.dict()
            self.param_list = [(comm, p) for p in self.param_list]
            
            if not AIRFLOW and self.using_tqdm: # first come first out. ie. unordered result
                for r in tqdm.tqdm(pool.imap_unordered(self.task_func, self.param_list), total=len(self.param_list), desc=self.tqdm_desc):
                    self.result.append(r)
            else: # ordered result
                for r in pool.map(self.task_func, self.param_list):
                    self.result.append(r)
                    
        return self.result
    
class mt():
    '''
    multi-threading interface.
    Good to use for io bound operation with shared connection obj.
    '''
    def __init__(self, task_func, param_list, n_process=4, using_tqdm=False, tqdm_desc='mt2'):
        self.task_func = task_func
        self.param_list = param_list
        self.n_process = n_process
        self.using_tqdm = using_tqdm
        self.tqdm_desc = tqdm_desc
        self.comm = {}
        
    def run(self):
        self.result = []
        #with ThreadPool(self.n_process) as pool, ThreadManager() as manager:
        with ThreadPool(self.n_process) as pool:
            # communication channel between thread / with parent
            # predefined codes
            # 1. 'break' : True - stop all children and return 
            self.param_list = [(self.comm, p) for p in self.param_list]
            
            if not AIRFLOW and self.using_tqdm: # first come first out. ie. unordered result
                for r in tqdm.tqdm(pool.imap_unordered(self.task_func, self.param_list), total=len(self.param_list), desc=self.tqdm_desc):
                    self.result.append(r)
            else: # ordered result
                for r in pool.map(self.task_func, self.param_list):
                    self.result.append(r)
                    
        return self.result

def n_split(len_list, task_load_min = 1, task_load_max = 100):
    '''
    multi_process의 task당 할당할 param의 수를 계산함
    len_list : 전체 param의 길이
    task_load_min : task당 처리하는 param의 최소 길이
    task_load_max : task당 처리하는 param의 최대 길이
    '''
    if len_list == 0:
        return 0
    
    assert task_load_min > 0 and task_load_max > 0 and task_load_min < task_load_max
    
    n_cpu = multiprocessing.cpu_count()
    
    if task_load_max * n_cpu < len_list:  # param의 충분히 크면 task마다 최대 길이의 param을 할당할 수 있도록 분할함
        return math.ceil(len_list / task_load_max)
    
    elif task_load_min * n_cpu < len_list:  # param이 적절하면 cpu 수만큼 분할함
        return n_cpu
    
    else: # param이 짧은 경우 task마다 최소 길이의 param을 할당할 수 있도록 분할함
        return math.ceil(len_list / task_load_min)
    
def in_query(iterable_param):
    return tuple(list(iterable_param)*2) if len(iterable_param) == 1 else tuple(iterable_param)

def id_range(id_start, id_end, idr_size): # return both side inclusive list of range(s)
    if id_start > id_end:
        raise Exception('Should be id_start <= id_end')
        return [[]]
    
    pl = [[i, i+idr_size-1] for i in range(id_start, id_end+1, idr_size)]
    pl[-1][-1] = id_end 
    
    return pl

def timer(func):
    """
        Print the runtime of the decorated function
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = datetime.now()#time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = datetime.now()#time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"{func.__name__!r} : {run_time}")
        return value
    return wrapper_timer

def log(func):
    """
        logging running time
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = datetime.now()#time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = datetime.now()#time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        
        os.makedirs('./log_runtime', exist_ok=True)
        with open(f'./log_runtime/{func.__name__}.pickle', 'ab') as f:
            pickle.dump({'func_name' : func.__name__, 'start_time' : start_time, 'end_time' : end_time, 'run_time':run_time}, f)
            
        return value
    return wrapper_timer

def read_log_runtime(function_name):
    log_file = f'./log_runtime/{function_name}.pickle'
    log = []
    with open(log_file, 'rb') as f:
        while True:
            try:
                log.append(pickle.load(f))
            except Exception as e:
                #print(e)
                break
                
    return pd.DataFrame(log)
