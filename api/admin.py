from django.contrib import admin
from . import models


@admin.register(models.Incident)
class IncidentAdmin(admin.ModelAdmin):
    list_display = ['start_time', 'end_time', 'video_player']
    list_filter = ('start_time', 'end_time')


@admin.register(models.CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ['email', 'phone_number', 'date_joined']
    list_filter = ('email', 'date_joined', 'phone_number')