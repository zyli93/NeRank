import torch
import numpy as np

from data_loader import DataLoader


dl = DataLoader(dataset="3dprinting", include_content=False)

qupos = np.array([771, 3771, 4771], dtype=np.int64)


v1func = np.vectorize(lambda x: dl.qtc(x))
v2func = np.vectorize(lambda x: np.array([[0, 1], [1,2]]))
inter2= v2func(qupos)
print(inter2)
print(type(inter2))


tensor = torch.FloatTensor(inter)


