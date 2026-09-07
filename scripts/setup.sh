#!/usr/bin/env bash
# Sıfırdan tam zincir: veri -> doğrulama -> özellikler -> segmentasyon -> model.
#
#   ./scripts/setup.sh              # config.yaml'daki tam veri seti (~1 dk)
#   ./scripts/setup.sh --fast       # 200 müşteri, hızlı duman testi
#   ./scripts/setup.sh --force      # üretilmiş veri varsa yeniden üret
#
# Komutları tek tek çalıştırmak da mümkün (README'de listeli); bu betik yalnızca
# sırayı ve ortam kontrolünü tek yerde toplar.

set -euo pipefail
cd "$(dirname "$0")/.."

HIZLI=""
ZORLA=0
for arg in "$@"; do
  case "$arg" in
    --fast)  HIZLI="--n-customers 200" ;;
    --force) ZORLA=1 ;;
    *) echo "Bilinmeyen seçenek: $arg" >&2; exit 2 ;;
  esac
done

source "$(dirname "$0")/_ortam.sh"
ortam_kontrol

adim() { printf '\n\033[1m▶ %s\033[0m\n' "$1"; }

# Yol config'ten okunur: KAMPANYA_CONFIG ile başka bir dizine yönlendirildiğinde
# sabit "data/raw" kontrolü yanlış dizine bakar ve üretimi hatalı biçimde atlar.
VERI_DOSYASI=$(python -c "from src.config import load_config, resolve_path; \
print(resolve_path(load_config()['data']['json']['raw_dir']) / 'customers.json')")

if [[ -f "$VERI_DOSYASI" && $ZORLA -eq 0 ]]; then
  echo "▶ Veri zaten var, üretim atlanıyor (yeniden üretmek için: --force)"
else
  adim "1/5  Sentetik veri üretiliyor"
  python -m src.data.generator $HIZLI
fi

adim "2/5  Veri kaynağı doğrulanıyor";  python -m src.data.validate
adim "3/5  Özellik tablosu";            python -m src.features.build
adim "4/5  Davranışsal segmentasyon";   python -m src.models.segmentation
adim "5/5  Kampanya kabul modeli";      python -m src.models.propensity

printf '\n\033[1mHazır.\033[0m Servisi başlatmak için:\n'
printf '  uvicorn src.api.main:app --reload      → http://127.0.0.1:8000/docs\n'
printf '  ./scripts/demo.sh                      → uçtan uca curl senaryosu\n\n'
