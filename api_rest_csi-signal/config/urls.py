from django.contrib import admin
from django.urls import path, include
from toy_front.views import index, historial_json

urlpatterns = [
    path("admin/", admin.site.urls),
    path("recognize/", include('reconocer.urls')),
    path("", index, name='dashboard'),
    path("api/historial/", historial_json, name='historial_json'),
]
