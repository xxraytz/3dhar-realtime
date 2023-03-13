import datetime
import os
import logging

import uuid

from django.db import models
from django.contrib import admin
from api.utils import send_email
from phone_field import PhoneField
from django.dispatch import receiver
from django.db.models.signals import (
    post_save
)

from django.utils.html import format_html
from web.settings import MEDIA_URL, LOCAL_HOST

# User creating
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from django.template.loader import render_to_string, get_template
from django.template import Context

from .managers import CustomUserManager

logger = logging.getLogger(__name__)


def get_upload_path(_, filename):
    _, ext = os.path.splitext(filename)
    return f'uploads/{uuid.uuid4()}{ext}'


class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(_('email'), unique=True)
    phone_number = PhoneField(default='7-910-167-42-63', help_text='Contact phone number')
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email


class Incident(models.Model):
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    real_video = models.FileField(upload_to=get_upload_path)  # validators=[FileExtensionValidator(['mp4'])]
    skeletal_video = models.FileField(upload_to=get_upload_path)  # validators=[FileExtensionValidator(['mp4'])]
    user = models.ForeignKey(
        CustomUser,
        related_name='incidents',
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    @admin.display(empty_value='???', description='Video')
    def video_player(self):
        """
        :return: videoview by html5
        """
        real_video_path = LOCAL_HOST + MEDIA_URL + str(self.real_video)
        skeletal_video_path = LOCAL_HOST + MEDIA_URL + str(self.skeletal_video)
        rendered = render_to_string('video-tag.html', {'real_video_path': real_video_path,
                                                       'skeletal_video_path': skeletal_video_path, })
        return format_html(format_string=rendered)

    class Meta:
        ordering = ('-created_at',)
        default_related_name = 'incidents'

    def __str__(self):
        return f'{self.start_time} - {self.end_time}'


@receiver(post_save, sender=Incident)
def incident_created(sender, instance, created, *args, **kwargs):
    if created:
        send_email(instance.real_video, instance.start_time.time().strftime('%H:%M'), instance.user.phone_number)
        logger.info("Created and send to email.")
    else:
        logger.info(f'Was just change this object: {instance.start_time!s}')