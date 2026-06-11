from django.shortcuts import render
from django.http import JsonResponse
from reconocer.models import CSIReading

def index(request):
    return render(request, 'toy_front/index.html')

def historial_json(request):
    qs = CSIReading.objects.all().order_by('-timestamp')[:100]
    data = [{
        "id": r.id,
        "ts": r.timestamp.isoformat(),
        "rssi": r.rssi,
        "pred": r.prediccion,
        "conf": r.confianza,
    } for r in qs]
    return JsonResponse(data, safe=False)
