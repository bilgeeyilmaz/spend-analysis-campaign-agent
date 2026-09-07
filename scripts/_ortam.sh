#!/usr/bin/env bash
# Ortak ortam kontrolü — setup.sh ve demo.sh bunu source eder.
# Anaconda'nın temel ortamı bu projeyi çalıştıramaz ama bunu geç ve anlaşılmaz
# biçimde söyler: temel ortamda Pydantic sürümü uygun olabilir, buna karşılık
# uvicorn hiç kurulu olmayabilir ve hata "No module named uvicorn" olarak
# sunucunun arka plan çıktısında kaybolur. O yüzden sürüm değil, projenin
# gerçekten ihtiyaç duyduğu modüllerin tamamı kontrol edilir.

ortam_kontrol() {
  python - <<'PY' || exit 1
import importlib.util
import sys

gerekli = ["pandas", "numpy", "sklearn", "pydantic", "yaml", "fastapi", "uvicorn", "joblib"]
eksik = [m for m in gerekli if importlib.util.find_spec(m) is None]

if eksik:
    sys.exit(
        f"HATA: şu modüller bulunamadı: {', '.join(eksik)}\n"
        f"      (kullanılan yorumlayıcı: {sys.executable})\n"
        "      Muhtemelen yanlış ortamdasınız:  conda activate kampanya\n"
        "      Ortam yoksa:  pip install -r requirements.txt"
    )

import pydantic
if int(pydantic.VERSION.split(".")[0]) < 2:
    sys.exit(f"HATA: Pydantic {pydantic.VERSION} bulundu, v2 gerekiyor "
             f"(field_validator/model_validate). conda activate kampanya")
PY
}
