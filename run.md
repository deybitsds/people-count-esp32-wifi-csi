Terminal 2 — ESP32 TX (emisor de paquetes WiFi):
```bash
source ~/esp/esp-idf/export.sh && cd /home/nando/programming/sistemas_embebidos/people-count-esp32-wifi-csi/material/esp-csi/examples/get-started/csi_send && idf.py -p /dev/ttyUSB0 monitor
```
Terminal 3 — Bridge (lee RX por serial y envía a la API):
```bash
python /home/nando/programming/sistemas_embebidos/people-count-esp32-wifi-csi/bridge_csi.py /dev/ttyUSB1 --server http://localhost:8000
```
