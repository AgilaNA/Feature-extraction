import itk


img_path = '/Users/agilana/Documents/GitHub/MIALab/data/train/212318/T2mni_biasfieldcorr_noskull.nii.gz'

inputPixelType = itk.ctype('unsigned short')
img = itk.imread(img_path, inputPixelType)
mask = itk.BinaryThresholdImageFilter.New(img)
mask.SetLowerThreshold(1)
mask.SetInsideValue(1)
mask_img = mask.GetOutput()

filtr = itk.CoocurrenceTextureFeaturesImageFilter.New(img)
filtr.SetMaskImage(mask)
filtr.SetNumberOfBinsPerAxis(10)
filtr.SetHistogramMinimum(0)
filtr.SetHistogramMaximum(2**16-1)
filtr.SetNeighborhoodRadius([5, 5, 5])


result = filtr.GetOutput()

itk.imwrite(result, "resultT2_212318.nrrd")
