#!/usr/bin/env python3
"""
CSI Bridge — Lee datos CSI del ESP32 RX por serial y los reenvía
via HTTP POST al servidor de conteo de personas.

Uso:
    python bridge_csi.py /dev/ttyUSB1
    python bridge_csi.py /dev/ttyUSB1 --server http://34.172.215.14:8691
"""

import sys, re, json, time, csv, argparse
from io import StringIO
from collections import deque

import serial
import requests


def parse_args():
    parser = argparse.ArgumentParser(description='CSI Bridge: ESP32 → HTTP POST')
    parser.add_argument('port', help='Puerto serial del ESP32 RX (ej: /dev/ttyUSB1)')
    parser.add_argument('--baud', type=int, default=921600, help='Baudrate (default: 921600)')
    parser.add_argument('--server', default='http://34.172.215.14:8691',
                        help='URL del servidor (default: http://34.172.215.14:8691)')
    parser.add_argument('--endpoint', default='/recognize/api/',
                        help='Endpoint POST (default: /recognize/api/)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Intervalo entre envios en segundos (default: 1.0)')
    parser.add_argument('--batch', type=int, default=5,
                        help='Paquetes a acumular antes de enviar (default: 5)')
    return parser.parse_args()


def parse_csi_line(line):
    """Parsea una línea CSI_DATA del monitor del ESP32.
    Formato: CSI_DATA,seq,mac,rssi,rate,sig_mode,...,rssi,...,noise_floor,...,len,first_word,"[data...]"
    Retorna (rssi, noise_floor, csi_values) o None si no es válida.
    """
    if 'CSI_DATA' not in line:
        return None

    reader = csv.reader(StringIO(line))
    try:
        row = next(reader)
    except StopIteration:
        return None

    if len(row) < 5:
        return None

    try:
        rssi = int(row[3])
    except (ValueError, IndexError):
        rssi = -100

    try:
        noise_floor = int(row[14])
    except (ValueError, IndexError):
        noise_floor = -100

    # La data CSI está en la última columna: "[n1 n2 n3 ...]"
    csi_str = row[-1]
    values = [int(x) for x in re.findall(r'-?\d+', csi_str)]
    if not values:
        return None

    return {
        'csi_values': values,
        'rssi': rssi,
        'noise_floor': noise_floor,
    }


def main():
    args = parse_args()
    url = args.server.rstrip('/') + args.endpoint
    buffer = deque()
    last_send = 0.0

    print(f"[CSI Bridge] Conectando a {args.port} a {args.baud} baud...")
    print(f"[CSI Bridge] Enviando a {url} cada {args.interval}s (batch={args.batch})")

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=1
    )

    print(f"[CSI Bridge] Conectado. Leyendo datos CSI...")

    sent_count = 0
    error_count = 0

    while True:
        try:
            raw = ser.readline().decode('utf-8', errors='replace').strip()
            if not raw:
                continue
        except serial.SerialException as e:
            print(f"[ERROR] Serial: {e}")
            time.sleep(2)
            continue

        parsed = parse_csi_line(raw)
        if parsed is None:
            continue

        buffer.append(parsed)
        now = time.time()

        if len(buffer) >= args.batch or (now - last_send) >= args.interval:
            if not buffer:
                continue

            # Promediar RSSI y noise_floor del batch
            avg_rssi = sum(p['rssi'] for p in buffer) // len(buffer)
            avg_nf = sum(p['noise_floor'] for p in buffer) // len(buffer)
            # Tomar el último CSI values del batch
            latest = buffer[-1]

            payload = {
                'csi_values': latest['csi_values'],
                'rssi': avg_rssi,
                'noise_floor': avg_nf,
            }

            try:
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    sent_count += 1
                    pred = data.get('prediccion_3clases', '?')
                    conf = data.get('confianza_3c', 0)
                    ts = time.strftime('%H:%M:%S')
                    print(f"[{ts}] → {pred} (conf:{conf:.2f}) rssi:{avg_rssi} batch:{len(buffer)}")
                else:
                    error_count += 1
                    print(f"[ERROR] HTTP {resp.status_code}: {resp.text[:100]}")
            except requests.exceptions.RequestException as e:
                error_count += 1
                print(f"[ERROR] Conexión: {e}")

            last_send = now
            buffer.clear()

            if sent_count % 10 == 0:
                print(f"[INFO] Enviados: {sent_count} | Errores: {error_count}")


if __name__ == '__main__':
    main()
