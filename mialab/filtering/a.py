import SimpleITK as sitk

path = '/Users/agilana/Documents/GitHub/MIALab/data/train/100307/T2mni_biasfieldcorr_noskull.nii.gz'
img = sitk.ReadImage(path)

sigmas = (10., 10., 10.)
edges = sitk.CannyEdgeDetection(sitk.Cast(img, sitk.sitkFloat32), 0.0, 0.0, sigmas)

sitk.WriteImage(edges, '/Users/agilana/Documents/GitHub/MIALab/data/train/100307/canny.nii.gz')
