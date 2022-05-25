from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
from firstPage.models import UploadImage
from django.http import FileResponse

import os
def resizeImage(Image):
    image = Image
    return image


def home(request):
    context = {'a' : 'HELLO NEW WORLD'}
    return render(request, 'home.html', context)

def breastCancer(request):
    context = {'pageValue' : 'Breast Cancer Detection'}
    return render(request, 'breastCancer.html', context)

def brainTumor(request):
    context = {'pageValue' : 'Brain Tumor'}
    return render(request, 'brainTumor.html', context)

def predictCancer(request):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    reloadModel = tf.keras.models.load_model('models/breast_cancer_pred.h5')
    if request.method == 'POST':
        temp={}
        temp['concavityMean']=request.POST.get('concavityMean')
        temp['concavePointsMean']=request.POST.get('concavePointsMean')
        temp['textureSe']=request.POST.get('textureSe')
        temp['smoothnessSe']=request.POST.get('smoothnessSe')
        temp['compactnessSe']=request.POST.get('compactnessSe')
        temp['concavitySe']=request.POST.get('concavitySe')
        temp['concavePointsSe']=request.POST.get('concavePointsSe')
        temp['symmetrySe']=request.POST.get('symmetrySe')
        temp['fractalDimensionSe']=request.POST.get('fractalDimensionSe')
        temp['concavityWorst']=request.POST.get('concavityWorst')
    input_data = list(temp.values())
    input_data = [float(i) for i in input_data]
    # input_data = (0.1689,0.06367,1.027,0.007405,0.04549,0.04588,0.01339,0.01738,0.004435,0.8402)

    #Converting to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    mean = [[0.06044219, 0.03590222, 1.09231332, 0.00638967, 0.01968866,
        0.02401426, 0.0099103 , 0.01863193, 0.00299774, 0.21344783]]

    scale = [[0.04900596, 0.02559238, 0.41563276, 0.00199298, 0.01061234,
        0.01508947, 0.0040322 , 0.00526381, 0.00130473, 0.14703332]]

    scaled_data = (input_data_reshaped - mean)/ scale

    prediction = reloadModel.predict(scaled_data)
    print(prediction)

    prediction_label = np.argmax(prediction)
    print(prediction_label)

    context = {'prediction': prediction_label}
    return render(request, 'cancerPrediction.html', context)



def predictTumor(request):
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import cv2 as cv
    reloadModel = tf.keras.models.load_model('models/brain_tumor_pred.h5')
    if request.method == "POST":
        fileUploaded = request.FILES['imgfile']
        imageName = request.FILES['imgfile'].name.split('/')[-1]
        document = UploadImage.objects.create(file = fileUploaded)
        document.save()
        tumorImage = UploadImage.objects.get(pk=document.id)
        imagePath = 'C:/finalYearProject/docspalsite/media/media/'+imageName
        temp_image = cv.imread(imagePath)
        resizedImage = cv.resize(temp_image, (224, 224))
        normalizedImage = np.array(resizedImage) / 255.0
        prediction = reloadModel.predict(np.array([normalizedImage],))
        prediction_label = np.argmax(prediction)
        print(prediction_label)
        context = {'prediction': prediction_label, 'image': tumorImage}
        return render(request, 'tumorDetection.html', context)


def test(request):
    context = {'pageValue' : 'Testing'}
    return render(request, 'testfile.html', context)