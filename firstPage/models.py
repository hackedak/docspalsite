from distutils.command.upload import upload
from django.db import models

class UploadImage(models.Model):
    file = models.ImageField(upload_to='media')
