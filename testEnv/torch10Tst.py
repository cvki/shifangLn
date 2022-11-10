import torch
import cv2
import numpy as np
from matplotlib import pyplot
import pandas as pd

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

path='/d2/code/gitRes/testEnv/rose.png'
rose=cv2.imread(path)
cv2.imshow('rose',rose)
cv2.waitKey(0)
cv2.destroyAllWindows()

