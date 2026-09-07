#!/usr/bin/env bash
# Uçtan uca demo: servisi başlatır, dört ucu sırayla çağırır, kapatır.
#
#   ./scripts/demo.sh              # varsayılan müşteri, özet çıktı
#   ./scripts/demo.sh C000042      # belirli bir müşteri
#   ./scripts/demo.sh --ham        # araç sonuçları dahil ham JSON
#
# GROQ_API_KEY yoksa sistem deterministik yedeğe düşer ve senaryo yine baştan
# sona çalışır — kapalı ağda demo yapabilmek bilinçli bir tasarım kararıdır.

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/_ortam.sh"
ortam_kontrol

MUSTERI="C000007"
HAM=0
for arg in "$@"; do
  case "$arg" in
    --ham) HAM=1 ;;
    *)     MUSTERI="$arg" ;;
  esac
done

PORT="${PORT:-8099}"
KOK="http://127.0.0.1:${PORT}"
SUNUCU_LOG="$(mktemp -t kampanya_demo)"

if [[ ! -f data/processed/customer_features.parquet ]]; then
  echo "HATA: özellik tablosu yok. Önce: ./scripts/setup.sh" >&2
  exit 1
fi

baslik() { printf '\n\033[1m── %s\033[0m\n' "$1"; }

# Ajan cevabının özeti; ham JSON için --ham.
ozet() {
  if [[ $HAM -eq 1 ]]; then
    python -m json.tool
  else
    python "$(dirname "$0")/_ozet.py"
  fi
}

# Sunucu çıktısı dosyaya: uygulama log satırları JSON'un ortasına karışmasın.
python -m uvicorn src.api.main:app --port "$PORT" --log-level warning > "$SUNUCU_LOG" 2>&1 &
SUNUCU=$!
trap 'kill $SUNUCU 2>/dev/null || true; rm -f "$SUNUCU_LOG"' EXIT

printf 'Servis açılıyor (veri ve model yükleniyor)'
HAZIR=0
for _ in $(seq 1 60); do
  if curl -s -m 2 "${KOK}/health" >/dev/null 2>&1; then HAZIR=1; break; fi
  if ! kill -0 "$SUNUCU" 2>/dev/null; then break; fi
  printf '.'; sleep 1
done

# Beklemenin sessizce "hazır" demesi, demoda en kötü hata biçimidir: sonraki
# adımlar boş cevabı ayrıştırmaya çalışıp anlamsız hata verir.
if [[ $HAZIR -eq 0 ]]; then
  printf '\n\nHATA: servis açılmadı. Sunucu çıktısı:\n' >&2
  tail -20 "$SUNUCU_LOG" >&2
  exit 1
fi
printf ' hazır\n'

baslik "1. GET /health — servis ve bağımlılıkları"
curl -s "${KOK}/health" | python -m json.tool

baslik "2. GET /customers/${MUSTERI}/profile — künye (PII içermez)"
curl -s "${KOK}/customers/${MUSTERI}/profile" | python -m json.tool

baslik "3. POST /recommend — önce analiz, sonra teklif"
curl -s -m 180 -X POST "${KOK}/recommend" -H 'Content-Type: application/json' \
  -d "{\"customer_id\":\"${MUSTERI}\"}" | ozet

baslik "4. POST /chat — serbest soru"
curl -s -m 180 -X POST "${KOK}/chat" -H 'Content-Type: application/json' \
  -d "{\"customer_id\":\"${MUSTERI}\",\"question\":\"Düzenli ödemelerim neler?\"}" | ozet

baslik "5. Hata yolu — olmayan müşteri (404 beklenir)"
curl -s -o /dev/null -w '  HTTP %{http_code}\n' -X POST "${KOK}/recommend" \
  -H 'Content-Type: application/json' -d '{"customer_id":"C999999"}'

printf '\n\033[1mDemo bitti.\033[0m Denetim izi: logs/audit.jsonl   (ham JSON için: --ham)\n\n'
