from django.db import models

class UploadImage(models.Model):
    file = models.FileField()
