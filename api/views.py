from django.shortcuts import get_object_or_404
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import Incident, CustomUser
from .serializers import IncidentSerializer, CustomUserSerializer
import logging

logger = logging.getLogger(__name__)


class IncidentViewSet(viewsets.ModelViewSet):
    queryset = Incident.objects.all()
    serializer_class = IncidentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        user_id = CustomUser.objects.get(email=request.user)
        data = request.data
        data['user'] = user_id.id
        serializer = IncidentSerializer(data=data)
        if serializer.is_valid():
            logger.info(serializer.validated_data)
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        logger.info('User: %s. New data was created.' % request.user)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CustomUserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer
    permission_classes = [IsAuthenticated]
