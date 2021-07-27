import numpy as np
import pandas as pd
import sys
import os
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def get_dir_dict(train_dir):
    patients = [train_dir + p for p in next(os.walk(train_dir))[1]]
    images = {
        "flair" : [p + "/FLAIR" for p in patients],
        "t1w": [p + "/T1w" for p in patients],
        "t1wce": [p + "/T1wCE" for p in patients],
        "t2w": [p + "/T2w" for p in patients]
    }
    return images

def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_dicom_series(path):
    """
    reads a dicom series (a list of files) represented by the input path
    
    args:
    path -- the path to the dicom folder
    """
    reader = sitk.ImageSeriesReader()
    fpaths = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(fpaths)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)

def resample_image(image, resample_spacing=None):
    """
    resamples a SimpleITK image
    """
    if resample_spacing == None:
        return image 
    else:
        # get pre voxel size and image size
        pre_vs = image.GetSpacing()
        pre_is = image.GetSize()
        
        # calculate post voxel size and
        post_is = [
            int(np.round(pre_is[0] * (pre_vs[0] / 2))),
            int(np.round(pre_is[1] * (pre_vs[1] / 2))),
            int(np.round(pre_is[2] * (pre_vs[2] / 2))) 
        ]
        
        return sitk.Resample(
            image1 = image,
            size = post_is,
            transform = sitk.Transform(),
            interpolator = sitk.sitkBSpline,
            outputOrigin = image.GetOrigin(),
            outputSpacing = resample_spacing,
            outputDirection = image.GetDirection(),
            defaultPixelValue = 0.0,
            outputPixelType = image.GetPixelID()
        )

def permute_norm(x):
    norm_hash =  x[1] + 2*x[2]
    return int(norm_hash)
def flip_norm(x):
    norm_hash = np.sum(x)
    if norm_hash > 0: 
        return False
    else:
        return True

def rotate_image(image):
    pre_flip_norms = np.round(image.GetDirection())
    pre_rot_norms = np.abs(pre_flip_norms)
    pre_rot_norms = pre_rot_norms.astype(int)
    
    # calculate the permutation for each normal vector and wrap in tuple
    post_rot_norms = [
        permute_norm(pre_rot_norms[0:3]),
        permute_norm(pre_rot_norms[3:6]),
        permute_norm(pre_rot_norms[6:9])
    ]    
    post_flip_norms = [
        flip_norm(pre_flip_norms[0:3]),
        flip_norm(pre_flip_norms[3:6]),
        flip_norm(pre_flip_norms[6:9])        
    ]
    
    image = sitk.PermuteAxes(image, post_rot_norms)
    image = sitk.Flip(image, post_flip_norms)
    
    return image

def n4_bias_correction(image, fit_level = 4, num_iterations = 50, hist_bins = 200):
    # if mask is not specified then set mask to default
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * fit_level)    
    corrector.SetNumberOfHistogramBins(hist_bins)
    
    return corrector.Execute(image, mask)

def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def process_image(series, resample_spacing=None):

    
    # read series -> rotate image -> resample image -> n4 bias correction
    # -> intensity normalization 
    image = read_dicom_series(series)
    image = rotate_image(image)
#     comment for testing
    image = resample_image(image, resample_spacing=resample_spacing)
    image = n4_bias_correction(image)
    image = sitk.RescaleIntensity(image, 0, 1)
    
    return image 

def map_safe_process(input_tuple):
        series, processed_dir, resample_spacing = input_tuple
        try:
            # construct path strings and make example folder
            out_series = os.path.basename(os.path.dirname(series))
            out_name = os.path.basename(series)
            out_file = processed_dir + out_series + "/" \
                        + os.path.basename(series) + ".nrrd"
            safe_make_dir(processed_dir + out_series)
            
            # process and write
            image = process_image(series, resample_spacing)
            sitk.WriteImage(image, out_file)
            
        except:
            # write series name to file if fail
            with open("./exceptions.txt", "a+") as f:
                f.write(series)
                f.write("/n")
            return False
        return True
    
def generate_imap_func(processed_dir, resample_spacing):
    return lambda x: map_safe_process(x, processed_dir, resample_spacing)