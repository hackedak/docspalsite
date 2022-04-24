from multiprocessing import context
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.



def index(request):
    context = {'a' : 'HELLO NEW WORLD'}
    return render(request, 'index.html', context)

def about(request):
    context = {'b' : 'HELLO about page'}
    return render(request, 'about.html', context)
