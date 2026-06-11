from django.urls import path
from .views import RecognizeViewSet

urlpatterns = [
    path('api/', RecognizeViewSet.as_view({'post': 'recognize'}), name='recognize_api'),
    path('historial/', RecognizeViewSet.as_view({'get': 'historial'}), name='historial_api'),
]
