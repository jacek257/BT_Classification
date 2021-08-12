import numpy as np
import pandas as pd
import sys
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def get_images(img_dir):
    patients = [img_dir + p for p in next(os.walk(img_dir))[1]]