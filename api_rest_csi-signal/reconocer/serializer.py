from rest_framework import serializers

class CSIReceiveSerializer(serializers.Serializer):
    csi_values = serializers.ListField(
        child=serializers.IntegerField(),
        allow_empty=False
    )
    rssi = serializers.IntegerField(required=False, default=-100)
    noise_floor = serializers.IntegerField(required=False, default=-100)
