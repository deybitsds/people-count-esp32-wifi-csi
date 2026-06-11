import os, re, traceback, json
import numpy as np
import joblib

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.viewsets import ViewSet
from rest_framework.permissions import AllowAny
from rest_framework.decorators import action

from .serializer import CSIReceiveSerializer
from .models import CSIReading

BASE = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR = os.path.join(BASE, 'modelo')

rf_bin = joblib.load(os.path.join(MODELO_DIR, 'rf_binario.pkl'))
rf_3c  = joblib.load(os.path.join(MODELO_DIR, 'rf_3clases.pkl'))
rf_8c  = joblib.load(os.path.join(MODELO_DIR, 'rf_8clases.pkl'))
scaler = joblib.load(os.path.join(MODELO_DIR, 'scaler.pkl'))

def parse_csi(arr):
    if isinstance(arr, str):
        arr = [int(x) for x in re.findall(r'-?\d+', arr)]
    return np.array(arr, dtype=np.int16)

def iq_to_amp(arr, start=12):
    iq = arr[start:]
    n = len(iq) // 2
    if n == 0:
        return np.array([], dtype=np.float32)
    return np.sqrt(iq[0::2].astype(np.float32)**2 + iq[1::2].astype(np.float32)**2)

def extract_features(amps, rssi, noise_floor):
    feats = []
    for sc in range(amps.shape[1]):
        c = amps[:, sc]
        feats += [c.mean(), c.std(), np.var(c), np.percentile(c,25), np.percentile(c,75), np.ptp(c)]
    feats += [amps.mean(), amps.std(), np.mean(np.std(amps, axis=0)), np.std(np.mean(amps, axis=0))]
    feats += [rssi, rssi * 0, noise_floor, noise_floor * 0]
    return np.array(feats).reshape(1, -1)

class RecognizeViewSet(ViewSet):
    permission_classes = [AllowAny]

    @action(detail=False, methods=['post'])
    def recognize(self, request):
        try:
            serializer = CSIReceiveSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            csi_raw = serializer.validated_data['csi_values']
            rssi = serializer.validated_data.get('rssi', -100)
            nf = serializer.validated_data.get('noise_floor', -100)

            arr = parse_csi(csi_raw)
            amp = iq_to_amp(arr)

            if len(amp) == 0:
                return Response({"error": "CSI data vacío"}, status=400)

            amps_2d = np.tile(amp, (5, 1))
            feats = extract_features(amps_2d, rssi, nf)

            n_cols = scaler.mean_.shape[0]
            if feats.shape[1] < n_cols:
                pad = np.zeros((1, n_cols - feats.shape[1]))
                feats = np.hstack([feats, pad])
            elif feats.shape[1] > n_cols:
                feats = feats[:, :n_cols]

            feats_s = scaler.transform(feats)

            p_bin = int(rf_bin.predict(feats_s)[0])
            p_3c  = int(rf_3c.predict(feats_s)[0])
            p_8c  = int(rf_8c.predict(feats_s)[0])

            prob_bin = float(rf_bin.predict_proba(feats_s).max())
            prob_3c  = float(rf_3c.predict_proba(feats_s).max())
            prob_8c  = float(rf_8c.predict_proba(feats_s).max())

            CSIReading.objects.create(
                rssi=rssi, noise_floor=nf,
                csi_raw=json.dumps(csi_raw) if isinstance(csi_raw, list) else str(csi_raw),
                prediccion=p_3c, confianza=prob_3c, modelo_usado='rf_3clases'
            )

            label_map = {0: "Vacío", 1: "1-3 personas", 2: "4-7 personas"}
            return Response({
                "prediccion_binaria": "Ocupado" if p_bin else "Vacío",
                "prediccion_3clases": label_map.get(p_3c, f"{p_3c}"),
                "prediccion_8clases": p_8c,
                "confianza_bin": round(prob_bin, 3),
                "confianza_3c": round(prob_3c, 3),
                "confianza_8c": round(prob_8c, 3),
                "status": 200,
            })

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

    @action(detail=False, methods=['get'])
    def historial(self, request):
        qs = CSIReading.objects.all().order_by('-timestamp')[:50]
        data = [{
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "rssi": r.rssi,
            "prediccion": r.prediccion,
            "confianza": r.confianza,
        } for r in qs]
        return Response(data)
