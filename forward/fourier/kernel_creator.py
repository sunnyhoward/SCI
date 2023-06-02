import numpy as np
import torch



def create_fourier_kernel():

    path_share = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'

    gratings_intensity_modulations = np.load(path_share+'gratings_intensity_modulations.npy')
    frames = np.load(path_share+'gratings_location_names.npy')
    wavelengths = np.load(path_share+'gratings_wavelengths.npy')


    # #horizontal dispersion in pixel per nm
    # shift_vertical_per_nm = 1.176
    # #horizontal dispersion in pixel at 800 nm
    # shift_vertical_800 = 929.6
    # #vertical dispersion in pixel per nm
    # shift_horizontal_per_nm = 0.865
    # #vertical dispersion in pixel at 800 nm
    # shift_horizontal_800 = 683.5


    # #minor horizontal dispersion in pixel per nm
    # shift_vertical_per_nm_minor = 0.009725072789688607
    # #minor vertical dispersion in pixel per nm
    # shift_horizontal_per_nm_minor = 0.00694456115609506

    # #minor horizontal dispersion in pixel at 800 nm
    # shift_vertical_800_minor = 10.213262024505468 + 2
    # #minor vertical dispersion in pixel at 800 nm
    # shift_horizontal_800_minor = -5.687984251728911

    #horizontal dispersion in pixel per nm
    shift_vertical_per_nm = 1.178928619271314
    #horizontal dispersion in pixel at 800 nm
    shift_vertical_800 = 929.8900351442538
    #vertical dispersion in pixel per nm
    shift_horizontal_per_nm = 0.8660647241600007
    #vertical dispersion in pixel at 800 nm
    shift_horizontal_800 = 683.5719160857097


    #minor horizontal dispersion in pixel per nm
    shift_vertical_per_nm_minor = 0.01128525750120758
    #minor vertical dispersion in pixel per nm
    shift_horizontal_per_nm_minor = 0.003468350809689265

    #minor horizontal dispersion in pixel at 800 nm
    shift_vertical_800_minor = 10.12119473261356
    #minor vertical dispersion in pixel at 800 nm
    shift_horizontal_800_minor = 2.586064353733273




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

        # print(wavelengths[i])
        
        # print(shift_instance_y)
        # print(center_x - int(np.floor(shift_instance_x)))
        # print((1-(shift_instance_x%1)) * intensity_polgrat_bottom[i])
        # print(center_x - int(np.ceil(shift_instance_x)))
        # print((shift_instance_x%1) * intensity_polgrat_bottom[i])
        # print('---')

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

        
    ninecopyconvolution = torch.fft.fftshift(ninecopyconvolution,dim=(0,1))

    ninecopyconvolution = ninecopyconvolution / torch.sum(ninecopyconvolution,dim=(0,1))
    return ninecopyconvolution