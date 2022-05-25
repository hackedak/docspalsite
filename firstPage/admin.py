from django.contrib import admin

# Register your models here.
from .models import UploadImage



class DocsPalAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)

admin.site.register(UploadImage, DocsPalAdmin)