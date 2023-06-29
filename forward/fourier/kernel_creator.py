import numpy as np
import torch
import os, sys
main_dir = os.path.dirname(os.path.abspath(''))
sys.path.insert(0, main_dir)
from models.helper import downsample_signal
from scipy.interpolate import interp1d
import torch.nn as nn



def create_fourier_kernel(desired_channels=21):

    path_share = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'

    gratings_intensity_modulations = np.load(path_share+'gratings_intensity_modulations_nm_resolution.npy')
    wavelengths = np.load(path_share+'gratings_wavelengths_nm_resolution.npy') #between 700 and 900 nm

    #    #vertical dispersion in pixel per nm
    shift_vertical_per_nm =  1.1324938301431489
    #vertical dispersion in pixel at 800 nm
    shift_vertical_800 = 929.1633879050196
    #horizontal dispersion in pixel per nm
    shift_horizontal_per_nm = 0.8429029601380736
    #horizontal dispersion in pixel at 800 nm
    shift_horizontal_800 = 683.6675760385812


    #minor vertical dispersion in pixel per nm
    shift_vertical_per_nm_minor = 0.041534067912511255
    #minor horizontal dispersion in pixel per nm
    shift_horizontal_per_nm_minor = 0.0005687614938061433
    #minor vertical dispersion in pixel at 800 nm
    shift_vertical_800_minor =  11.33549523982135
    #minor horizontal dispersion in pixel at 800 nm
    shift_horizontal_800_minor = 1.9162529524255756



    size_x = 2048
    size_y = 2448


    intensity_zero_order = gratings_intensity_modulations[4]
    intensity_polgrat_bottom = gratings_intensity_modulations[7]
    intensity_polgrat_top = gratings_intensity_modulations[1]
    intensity_grat_left = gratings_intensity_modulations[3]
    intensity_grat_right = gratings_intensity_modulations[5]
    intensity_top_left = gratings_intensity_modulations[0]
    intensity_top_right = gratings_intensity_modulations[2]
    intensity_bottom_left = gratings_intensity_modulations[6]
    intensity_bottom_right = gratings_intensity_modulations[8]



    center_x = size_x//2
    center_y = size_y//2
    ninecopyconvolution = torch.zeros([size_x, size_y, len(wavelengths)])#.to(device)

    for i in np.arange(len(wavelengths)):
        additional_shift_vertical = (wavelengths[i] - 800) * shift_vertical_per_nm
        additional_shift_horizontal = (wavelengths[i] - 800) * shift_horizontal_per_nm
        additional_shift_vertical_minor = (wavelengths[i] - 800) * shift_vertical_per_nm_minor
        additional_shift_horizontal_minor = (wavelengths[i] - 800) * shift_horizontal_per_nm_minor
        shift_instance_y = (shift_vertical_800+additional_shift_vertical)
        shift_instance_x = (shift_horizontal_800+additional_shift_horizontal)
        shift_instance_y_minor = (shift_vertical_800_minor+additional_shift_vertical_minor)
        shift_instance_x_minor = (shift_horizontal_800_minor+additional_shift_horizontal_minor)

        ninecopyconvolution[center_x, center_y, i] += intensity_zero_order[i]
        
        ninecopyconvolution[center_x - int(np.floor(shift_instance_x)), center_y + int(np.floor(shift_instance_y_minor)), i] += (1-(shift_instance_x%1)) * (1-(shift_instance_y_minor%1)) * intensity_polgrat_bottom[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x)), center_y + int(np.floor(shift_instance_y_minor)), i] += (shift_instance_x%1) * (1-(shift_instance_y_minor%1)) * intensity_polgrat_bottom[i]
        ninecopyconvolution[center_x - int(np.floor(shift_instance_x)), center_y + int(np.ceil(shift_instance_y_minor)), i] += (1-(shift_instance_x%1)) * (shift_instance_y_minor%1) * intensity_polgrat_bottom[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x)), center_y + int(np.ceil(shift_instance_y_minor)), i] += (shift_instance_x%1) * (shift_instance_y_minor%1) * intensity_polgrat_bottom[i]


        ninecopyconvolution[center_x + int(np.floor(shift_instance_x)), center_y - int(np.floor(shift_instance_y_minor)), i] += (1-(shift_instance_x%1)) * (1-(shift_instance_y_minor%1)) * intensity_polgrat_top[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x)), center_y - int(np.floor(shift_instance_y_minor)), i] += (shift_instance_x%1) * (1-(shift_instance_y_minor%1)) * intensity_polgrat_top[i]
        ninecopyconvolution[center_x + int(np.floor(shift_instance_x)), center_y - int(np.ceil(shift_instance_y_minor)), i] += (1-(shift_instance_x%1)) * (shift_instance_y_minor%1) * intensity_polgrat_top[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x)), center_y - int(np.ceil(shift_instance_y_minor)), i] += (shift_instance_x%1) * (shift_instance_y_minor%1) * intensity_polgrat_top[i]

        ninecopyconvolution[center_x + int(np.floor(shift_instance_x_minor)), center_y - int(np.floor(shift_instance_y)), i] += (1-(shift_instance_x_minor%1)) * (1-(shift_instance_y%1)) * intensity_grat_left[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x_minor)), center_y - int(np.floor(shift_instance_y)), i] += (shift_instance_x_minor%1) * (1-(shift_instance_y%1)) * intensity_grat_left[i]
        ninecopyconvolution[center_x + int(np.floor(shift_instance_x_minor)), center_y - int(np.ceil(shift_instance_y)), i] += (1-(shift_instance_x_minor%1)) * (shift_instance_y%1) * intensity_grat_left[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x_minor)), center_y - int(np.ceil(shift_instance_y)), i] += (shift_instance_x_minor%1) * (shift_instance_y%1) * intensity_grat_left[i]

        ninecopyconvolution[center_x - int(np.floor(shift_instance_x_minor)), center_y + int(np.floor(shift_instance_y)), i] += (1-(shift_instance_x_minor%1)) * (1-(shift_instance_y%1)) * intensity_grat_right[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x_minor)), center_y + int(np.floor(shift_instance_y)), i] += (shift_instance_x_minor%1) * (1-(shift_instance_y%1)) * intensity_grat_right[i]
        ninecopyconvolution[center_x - int(np.floor(shift_instance_x_minor)), center_y + int(np.ceil(shift_instance_y)), i] += (1-(shift_instance_x_minor%1)) * (shift_instance_y%1) * intensity_grat_right[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x_minor)), center_y + int(np.ceil(shift_instance_y)), i] += (shift_instance_x_minor%1) * (shift_instance_y%1) * intensity_grat_right[i]

        ninecopyconvolution[center_x - int(np.floor(shift_instance_x-shift_instance_x_minor)), center_y - int(np.floor(shift_instance_y-shift_instance_y_minor)), i] += (1-((shift_instance_x-shift_instance_x_minor)%1)) * (1-((shift_instance_y-shift_instance_y_minor)%1)) * intensity_bottom_left[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x-shift_instance_x_minor)), center_y - int(np.floor(shift_instance_y-shift_instance_y_minor)), i] += ((shift_instance_x-shift_instance_x_minor)%1) * (1-((shift_instance_y-shift_instance_y_minor)%1)) * intensity_bottom_left[i]
        ninecopyconvolution[center_x - int(np.floor(shift_instance_x-shift_instance_x_minor)), center_y - int(np.ceil(shift_instance_y-shift_instance_y_minor)), i] += (1-((shift_instance_x-shift_instance_x_minor)%1)) * ((shift_instance_y-shift_instance_y_minor)%1) * intensity_bottom_left[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x-shift_instance_x_minor)), center_y - int(np.ceil(shift_instance_y-shift_instance_y_minor)), i] += ((shift_instance_x-shift_instance_x_minor)%1) * ((shift_instance_y-shift_instance_y_minor)%1) * intensity_bottom_left[i]

        ninecopyconvolution[center_x - int(np.floor(shift_instance_x+shift_instance_x_minor)), center_y + int(np.floor(shift_instance_y+shift_instance_y_minor)), i] += (1-((shift_instance_x+shift_instance_x_minor)%1)) * (1-((shift_instance_y+shift_instance_y_minor)%1)) * intensity_top_left[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x+shift_instance_x_minor)), center_y + int(np.floor(shift_instance_y+shift_instance_y_minor)), i] += ((shift_instance_x+shift_instance_x_minor)%1) * (1-((shift_instance_y+shift_instance_y_minor)%1)) * intensity_top_left[i]
        ninecopyconvolution[center_x - int(np.floor(shift_instance_x+shift_instance_x_minor)), center_y + int(np.ceil(shift_instance_y+shift_instance_y_minor)), i] += (1-((shift_instance_x+shift_instance_x_minor)%1)) * ((shift_instance_y+shift_instance_y_minor)%1) * intensity_top_left[i]
        ninecopyconvolution[center_x - int(np.ceil(shift_instance_x+shift_instance_x_minor)), center_y + int(np.ceil(shift_instance_y+shift_instance_y_minor)), i] += ((shift_instance_x+shift_instance_x_minor)%1) * ((shift_instance_y+shift_instance_y_minor)%1) * intensity_top_left[i]

        ninecopyconvolution[center_x + int(np.floor(shift_instance_x+shift_instance_x_minor)), center_y - int(np.floor(shift_instance_y+shift_instance_y_minor)), i] += (1-((shift_instance_x+shift_instance_x_minor)%1)) * (1-((shift_instance_y+shift_instance_y_minor)%1)) * intensity_bottom_right[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x+shift_instance_x_minor)), center_y - int(np.floor(shift_instance_y+shift_instance_y_minor)), i] += ((shift_instance_x+shift_instance_x_minor)%1) * (1-((shift_instance_y+shift_instance_y_minor)%1)) * intensity_bottom_right[i]
        ninecopyconvolution[center_x + int(np.floor(shift_instance_x+shift_instance_x_minor)), center_y - int(np.ceil(shift_instance_y+shift_instance_y_minor)), i] += (1-((shift_instance_x+shift_instance_x_minor)%1)) * ((shift_instance_y+shift_instance_y_minor)%1) * intensity_bottom_right[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x+shift_instance_x_minor)), center_y - int(np.ceil(shift_instance_y+shift_instance_y_minor)), i] += ((shift_instance_x+shift_instance_x_minor)%1) * ((shift_instance_y+shift_instance_y_minor)%1) * intensity_bottom_right[i]

        ninecopyconvolution[center_x + int(np.floor(shift_instance_x-shift_instance_x_minor)), center_y + int(np.floor(shift_instance_y-shift_instance_y_minor)), i] += (1-((shift_instance_x-shift_instance_x_minor)%1)) * (1-((shift_instance_y-shift_instance_y_minor)%1)) * intensity_top_right[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x-shift_instance_x_minor)), center_y + int(np.floor(shift_instance_y-shift_instance_y_minor)), i] += ((shift_instance_x-shift_instance_x_minor)%1) * (1-((shift_instance_y-shift_instance_y_minor)%1)) * intensity_top_right[i]
        ninecopyconvolution[center_x + int(np.floor(shift_instance_x-shift_instance_x_minor)), center_y + int(np.ceil(shift_instance_y-shift_instance_y_minor)), i] += (1-((shift_instance_x-shift_instance_x_minor)%1)) * ((shift_instance_y-shift_instance_y_minor)%1) * intensity_top_right[i]
        ninecopyconvolution[center_x + int(np.ceil(shift_instance_x-shift_instance_x_minor)), center_y + int(np.ceil(shift_instance_y-shift_instance_y_minor)), i] += ((shift_instance_x-shift_instance_x_minor)%1) * ((shift_instance_y-shift_instance_y_minor)%1) * intensity_top_right[i]


    ninecopyconvolution = torch.tensor(downsample_signal(ninecopyconvolution.numpy(), desired_channels=desired_channels, initial_range=[700,900], desired_range=[750,850], interp_axis=-1, interp_type='average')).float()


    ninecopyconvolution = ninecopyconvolution / torch.sum(ninecopyconvolution,dim=(0,1))
    return ninecopyconvolution.unsqueeze(0).permute(0,3,1,2)