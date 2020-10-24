import torch
torch.nn.modules.utils._pair()
# data = torch.tensor([101, 102, 103, 104])
# index = torch.tensor([1, 0, 3, 2, 1, 0],)
# res = torch.gather(data, 0, index)
# print(res)
def log(func):
    print('.....')
    # func()
    def wrapper():
        print('log开始 ...')
        # func()
        print('log结束 ...')
        return 'kkk'

    # return wrapper
    return wrapper


@log
def test():
    print('test ..')


print(test())

