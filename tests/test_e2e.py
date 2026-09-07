"""Uçtan uca zincir testi — Gün 10 kontrol noktasının otomatik hâli.

Diğer testler katmanları **fonksiyon çağırarak** sınar; burada zincir gerçek
komut satırı araçlarıyla, ayrı süreçlerde, sıfırdan çalıştırılır:

    üreteç -> doğrulayıcı -> özellik tablosu -> segmentasyon -> propensity -> API

Bunun fonksiyon seviyesindeki testlerle yakalanamayan üç şeyi vardır:

  1. **`__main__` yolları.** `python -m src.features.build` çalışmıyorsa hiçbir
     birim testi kırmızıya dönmez; README'deki komut ise çalışmaz.
  2. **Adımlar arası sözleşme.** Her adım bir öncekinin diske yazdığını okur.
     Parquet kolon adı değişirse hata üç katman sonra, model eğitiminde çıkar.
  3. **Servisin gerçek açılışı.** `tests/test_api.py` lifespan'i bilerek
     atlar (80 MB veri okumamak için); ısınma yolu yalnızca burada çalışır.

Zincir, `KAMPANYA_CONFIG` ile geçici bir dizine yönlendirilir — testin gerçek
`data/` ve `models/` içeriğine dokunmadığı ayrıca doğrulanır.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest
import yaml

from src.config import CONFIG_ENV, PROJECT_ROOT, load_config

N_CUSTOMERS = 220

#: API adımı ayrı süreçte koşar: `load_config()` lru_cache'li olduğu için aynı
#: süreçte iki farklı konfigürasyonla çalışmak mümkün değil.
API_KODU = """
import json
from fastapi.testclient import TestClient
from src.api.main import app

with TestClient(app) as istemci:                 # lifespan + ısınma burada çalışır
    saglik = istemci.get("/health").json()
    musteriler = app.state.agent.ctx.source.get_customers()
    cid = musteriler[musteriler["opt_in_marketing"]].iloc[0]["customer_id"]
    oneri = istemci.post("/recommend", json={"customer_id": cid})
    profil = istemci.get(f"/customers/{cid}/profile")

print("__SONUC__" + json.dumps({
    "saglik": saglik,
    "customer_id": cid,
    "oneri_kodu": oneri.status_code,
    "oneri": oneri.json(),
    "profil_kodu": profil.status_code,
}, ensure_ascii=False))
"""


def _kosla(env: dict, *komut: str) -> str:
    sonuc = subprocess.run(
        [sys.executable, *komut], env=env, cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=900,
    )
    assert sonuc.returncode == 0, (
        f"komut düştü: {' '.join(komut)}\n"
        f"--- stdout ---\n{sonuc.stdout[-3000:]}\n--- stderr ---\n{sonuc.stderr[-3000:]}"
    )
    return sonuc.stdout


#: Zincirin ezmemesi gereken gerçek çıktılar.
KORUNAN_YOLLAR = [
    "data/raw/customers.json",
    "data/raw/transactions.json",
    "data/processed/customer_features.parquet",
    "models/propensity.joblib",
    "models/segmentation.joblib",
]


def _parmak_izi() -> dict[str, tuple[int, float] | None]:
    """Korunan dosyaların (boyut, değişim zamanı) çifti; yoksa None."""
    izler = {}
    for yol in KORUNAN_YOLLAR:
        p = PROJECT_ROOT / yol
        izler[yol] = (p.stat().st_size, p.stat().st_mtime) if p.exists() else None
    return izler


@pytest.fixture(scope="module")
def zincir(tmp_path_factory) -> dict:
    """Bütün zinciri bir kez çalıştırır; testler çıktılarını okur."""
    kok = tmp_path_factory.mktemp("e2e")

    cfg = copy.deepcopy(load_config())
    cfg["data"]["json"]["raw_dir"] = str(kok / "raw")
    cfg["data"]["processed_dir"] = str(kok / "processed")
    cfg["models"]["dir"] = str(kok / "models")
    cfg["audit"]["log_path"] = str(kok / "logs" / "audit.jsonl")
    cfg["generator"]["n_customers"] = N_CUSTOMERS
    cfg["generator"]["seed"] = 7
    cfg["agent"]["provider"] = "mock"
    # Izgara taraması burada ölçülmüyor; tek kombinasyon zinciri hızlı tutar.
    cfg["models"]["propensity"]["param_grid"] = {
        "learning_rate": [0.05], "num_leaves": [4], "n_estimators": [100],
    }

    (kok / "raw").mkdir()
    # Katalog elle yazılmış ve versiyonlanmış veridir; üreteç onu üretmez.
    (kok / "raw" / "campaigns.json").write_text(
        (PROJECT_ROOT / "data" / "raw" / "campaigns.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cfg_path = kok / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    env = {**os.environ, CONFIG_ENV: str(cfg_path), "GROQ_API_KEY": ""}

    # Zincir öncesi gerçek çıktıların parmak izi; yalıtım testi bunu karşılaştırır.
    onceki = _parmak_izi()

    ciktilar = {
        "generator": _kosla(env, "-m", "src.data.generator"),
        "validate": _kosla(env, "-m", "src.data.validate"),
        "features": _kosla(env, "-m", "src.features.build"),
        "segmentation": _kosla(env, "-m", "src.models.segmentation"),
        "propensity": _kosla(env, "-m", "src.models.propensity"),
        "api": _kosla(env, "-c", API_KODU),
    }
    ciktilar["kok"] = kok
    ciktilar["onceki_izler"] = onceki
    ciktilar["sonraki_izler"] = _parmak_izi()
    ciktilar["api_sonuc"] = json.loads(ciktilar["api"].split("__SONUC__")[1])
    return ciktilar


# --------------------------------------------------------------------------


def test_zincir_bastan_sona_calisir(zincir):
    """Beş komut da sıfır dönmeli ve beklenen dosyaları yazmalı."""
    kok = zincir["kok"]
    beklenen = [
        kok / "raw" / "customers.json",
        kok / "raw" / "transactions.json",
        kok / "raw" / "interactions.json",
        kok / "processed" / "customer_features.parquet",
        kok / "processed" / "customer_segments.parquet",
        kok / "models" / "segmentation.joblib",
        kok / "models" / "propensity.joblib",
    ]
    eksik = [p.name for p in beklenen if not p.exists()]
    assert not eksik, f"zincir bu dosyaları yazmadı: {eksik}"


def test_dogrulayici_uretilen_veriyi_onayliyor(zincir):
    """Üreteç ile şema arasındaki sözleşme: biri değişip diğeri değişmezse burada patlar."""
    assert "BAŞARILI" in zincir["validate"]


def test_varsayilan_algoritma_ile_egitiliyor(zincir):
    """`--algorithm hist` bayrağı OLMADAN eğitim.

    LightGBM'i bloke eden libomp sorunu geri gelirse (ya da config'in varsayılanı
    çalışmaz hâle gelirse) README'deki komut bozulur; testin görmesi gereken yer burası.
    """
    assert "algoritma            : lightgbm" in zincir["propensity"]
    assert "AUC" in zincir["propensity"]


def test_segmentasyon_kullanilabilir_segment_uretiyor(zincir):
    """Mikro küme reddi zincirin sonunda da geçerli (küçük veri setinde bile)."""
    assert "seçilen k" in zincir["segmentation"]


def test_api_gercek_acilisla_ayaga_kalkiyor(zincir):
    """Lifespan + ısınma: yalnızca bu test gerçekten çalıştırır."""
    saglik = zincir["api_sonuc"]["saglik"]

    assert saglik["status"] == "ok", saglik
    assert all(d["hazir"] for d in saglik["bagimliliklar"].values())


def test_api_sifirdan_uretilen_veriyle_oneri_veriyor(zincir):
    """Kontrol noktasının özü: taze veri -> taze model -> HTTP üzerinden cevap."""
    sonuc = zincir["api_sonuc"]

    assert sonuc["oneri_kodu"] == 200
    assert sonuc["profil_kodu"] == 200
    assert sonuc["oneri"]["answer"].strip()
    assert sonuc["oneri"]["tool_calls"]
    assert not sonuc["oneri"]["uyarilar"], sonuc["oneri"]["uyarilar"]


def test_zincir_gercek_veri_dizinine_dokunmuyor(zincir):
    """`KAMPANYA_CONFIG` yalıtımının kendisinin testi.

    Yolların farklı olduğunu iddia etmek yetmez (o zaten totolojidir); zincir
    öncesi ve sonrası gerçek dosyaların boyut/değişim zamanı karşılaştırılır.
    Yalıtım bozulursa test paketi kullanıcının üretilmiş verisini sessizce ezer
    ve bu ancak veri yeniden üretilmek istendiğinde fark edilir.
    """
    # Yol adı yanlış yazılırsa parmak izi hep None olur ve test yine "geçer";
    # en az bir gerçek dosyanın izlendiğini şart koşuyoruz.
    assert any(iz is not None for iz in zincir["onceki_izler"].values()), (
        "korunan yolların hiçbiri mevcut değil — test boşa dönüyor olabilir"
    )
    degisen = [
        yol for yol in KORUNAN_YOLLAR
        if zincir["onceki_izler"][yol] != zincir["sonraki_izler"][yol]
    ]
    assert not degisen, f"zincir gerçek dosyaları değiştirdi: {degisen}"
