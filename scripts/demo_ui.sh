#!/usr/bin/env bash
# API + Streamlit arayüzünü birlikte başlatır (Ctrl-C ikisini de kapatır).
#
#   ./scripts/demo_ui.sh
#
# Arayüz servisin istemcisidir; iki süreç gerekir. Tek komutla başlatmanın sebebi
# demo günü "API'yi açmayı unuttum" hatasını mümkün kılmamak.

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/_ortam.sh"
ortam_kontrol

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

if [[ ! -f data/processed/customer_features.parquet ]]; then
  echo "HATA: özellik tablosu yok. Önce: ./scripts/setup.sh" >&2
  exit 1
fi

python -m uvicorn src.api.main:app --port "$API_PORT" --log-level warning &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

printf 'API açılıyor (veri ve model yükleniyor)'
HAZIR=0
for _ in $(seq 1 60); do
  if curl -s -m 2 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then HAZIR=1; break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then break; fi
  printf '.'; sleep 1
done
if [[ $HAZIR -eq 0 ]]; then
  printf '\nHATA: API açılmadı.\n' >&2
  exit 1
fi
printf ' hazır\n\n'

streamlit run ui/streamlit_app.py \
  --server.port "$UI_PORT" \
  --server.address 127.0.0.1
