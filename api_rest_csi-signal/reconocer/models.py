from django.db import models

class CSIReading(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    rssi = models.IntegerField(null=True, blank=True)
    noise_floor = models.IntegerField(null=True, blank=True)
    csi_raw = models.TextField()
    prediccion = models.IntegerField(null=True, blank=True)
    confianza = models.FloatField(null=True, blank=True)
    modelo_usado = models.CharField(max_length=20, default='rf_3clases')
