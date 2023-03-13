import logging
from django.core.validators import FileExtensionValidator
from rest_framework import serializers
from api.models import Incident, CustomUser

logger = logging.getLogger(__name__)


class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = '__all__'


class IncidentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Incident
        video = serializers.FileField()
        fields = ('start_time', 'end_time', 'user', 'real_video', 'skeletal_video')

    def validate(self, data):
        if data['start_time'] > data['end_time']:
            raise serializers.ValidationError("Incorrect timestamps")
        #  Check video validation
        video_valid = FileExtensionValidator(['mp4'])
        video_valid(data['real_video'])
        video_valid(data['skeletal_video'])

        return data

    def create(self, validated_data):
        logger.info(validated_data)
        created_data = Incident.objects.create(**validated_data)
        return created_data
