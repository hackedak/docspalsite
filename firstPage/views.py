from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

import joblib
import pandas as pd
import numpy as np

reloadModel=joblib.load('models/trained_model.sav')


def index(request):
    context = {'a' : 'HELLO NEW WORLD'}
    return render(request, 'index.html', context)

def breastCancer(request):
    context = {'pageValue' : 'Breast Cancer Detection'}
    return render(request, 'breastCancer.html', context)

def liverSegment(request):
    context = {'pageValue' : 'Liver Segmentation'}
    return render(request, 'liverSegment.html', context)

def predictCancer(request):
    if request.method == 'POST':
        temp={}
        temp['radiusMean']=request.POST.get('radiusMean')
        temp['textureMean']=request.POST.get('textureMean')
        temp['perimeterMean']=request.POST.get('perimeterMean')
        temp['areaMean']=request.POST.get('areaMean')
        temp['smoothnessMean']=request.POST.get('smoothnessMean')
        temp['compactnessMean']=request.POST.get('compactnessMean')
        temp['concavityMean']=request.POST.get('concavityMean')
        temp['concavePointsMean']=request.POST.get('concavePointsMean')
        temp['symmetryMean']=request.POST.get('symmetryMean')
        temp['fractalDimensionMean']=request.POST.get('fractalDimensionMean')
        temp['radiusSe']=request.POST.get('radiusSe')
        temp['textureSe']=request.POST.get('textureSe')
        temp['perimeterSe']=request.POST.get('perimeterSe')
        temp['areaSe']=request.POST.get('areaSe')
        temp['smoothnessSe']=request.POST.get('smoothnessSe')
        temp['compactnessSe']=request.POST.get('compactnessSe')
        temp['concavitySe']=request.POST.get('concavitySe')
        temp['concavePointsSe']=request.POST.get('concavePointsSe')
        temp['symmetrySe']=request.POST.get('symmetrySe')
        temp['fractalDimensionSe']=request.POST.get('fractalDimensionSe')
        temp['radiusWorst']=request.POST.get('radiusWorst')
        temp['textureWorst']=request.POST.get('textureWorst')
        temp['perimeterWorst']=request.POST.get('perimeterWorst')
        temp['areaWorst']=request.POST.get('areaWorst')
        temp['smoothnessWorst']=request.POST.get('smoothnessWorst')
        temp['compactnessWorst']=request.POST.get('compactnessWorst')
        temp['concavityWorst']=request.POST.get('concavityWorst')
        temp['concavePointsWorst']=request.POST.get('concavePointsWorst')
        temp['symmetryWorst']=request.POST.get('symmetryWorst')
        temp['fractalDimensionWorst']=request.POST.get('fractalDimensionWorst')

    # inputData = pd.DataFrame({'x':temp}).transpose()
    input_data = list(temp.values())
    print(input_data)
    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = reloadModel.predict(input_data_reshaped)[0]
    # scoreVal = reloadModel.predict(inputData)[0]
    context = {'prediction' : prediction}
    return render(request, 'breastCancer.html', context)



def test(request):
    context = {'pageValue' : 'Testing'}
    return render(request, 'testfile.html', context)