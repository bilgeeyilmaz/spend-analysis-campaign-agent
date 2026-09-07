"""Özellik tablosu ve davranışsal segmentasyon testleri.

Bu katmanın testleri iki soruya cevap verir:

  1. **Uyum** — modele girmemesi gereken alan gerçekten girmiyor mu? Bu bir
     kod kalitesi meselesi değil, projenin hukuki iddiasının kanıtıdır.
  2. **Tutarlılık** — paylar toplamı 1 mi, NaN sızıyor mu, `top_category`
     gerçekten en büyük pay mı? Sessiz bozulmalar burada yakalanır, üç katman
     sonra modelin metriğinde değil.

Testler kendi verisini üretir (`data/raw/` altındaki gerçek veriye dokunmaz).
"""

from __future__ import annotations

import copy
import json
import shutil

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data import generator
from src.features.build import (
    BEHAVIORAL_FEATURES,
    EXCLUDED_FEATURES,
    LEGAL_GATE_COLUMNS,
    build_feature_table,
    model_feature_columns,
)
from src.features.categories import CATEGORIES, CHANNELS
from src.models.segmentation import MIN_KUME_ORANI, segment_et

N_CUSTOMERS = 220


@pytest.fixture(scope="module")
def config_ve_tablo(tmp_path_factory) -> tuple[dict, pd.DataFrame]:
    """Küçük bir veri seti üretip özellik tablosunu kurar."""
    out_dir = tmp_path_factory.mktemp("raw")
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "campaigns.json", out_dir / "campaigns.json")
    assert generator.main([
        "--n-customers", str(N_CUSTOMERS), "--out", str(out_dir),
        "--as-of", "2026-08-13", "--seed", "7",
    ]) == 0

    config = copy.deepcopy(load_config())
    config["data"]["json"]["raw_dir"] = str(out_dir)
    return config, build_feature_table(config)


@pytest.fixture(scope="module")
def tablo(config_ve_tablo) -> pd.DataFrame:
    return config_ve_tablo[1]


# --------------------------------------------------------------------------
# Uyum: modele girmemesi gerekenler
# --------------------------------------------------------------------------


def test_gender_ozellik_tablosunda_yok(tablo):
    """Cinsiyete göre farklılaşan teklif hukuki risk taşır; alan tabloya hiç girmez."""
    assert "gender" not in tablo.columns
    assert "gender" in EXCLUDED_FEATURES


def test_dislanan_alanlar_tabloda_yok(tablo):
    for alan in EXCLUDED_FEATURES:
        assert alan not in tablo.columns, f"{alan} özellik tablosuna sızmış"


def test_opt_in_tabloda_var_ama_model_girdisi_degil(tablo):
    """Yasal kapı: uygunluk kontrolü okusun diye tabloda, modelde değil.

    Modele girerse "izinsiz müşteri zaten kabul etmiyor" korelasyonu öğrenilir
    ve izin, bir skora dönüşür — oysa izin skordan bağımsız, ondan önce gelir.
    """
    assert "opt_in_marketing" in tablo.columns
    assert "opt_in_marketing" in LEGAL_GATE_COLUMNS
    assert "opt_in_marketing" not in model_feature_columns(tablo)


def test_model_girdisinde_metin_kolon_yok(tablo):
    """Sayısal olmayan alanlar (customer_id, top_category) modele gitmemeli."""
    girdiler = model_feature_columns(tablo)
    assert "customer_id" not in girdiler
    assert "top_category" not in girdiler


# --------------------------------------------------------------------------
# Tutarlılık
# --------------------------------------------------------------------------


def test_her_musteri_icin_tek_satir(config_ve_tablo):
    config, tablo = config_ve_tablo
    assert len(tablo) == N_CUSTOMERS
    assert tablo["customer_id"].is_unique


def test_kategori_paylari_toplami_bir(tablo):
    paylar = tablo[[f"cat_share_{c}" for c in CATEGORIES]].sum(axis=1)
    assert paylar.between(0.999, 1.001).all(), "kategori payları 1'e toplanmıyor"


def test_kanal_paylari_toplami_bir(tablo):
    paylar = tablo[[f"channel_share_{c}" for c in CHANNELS]].sum(axis=1)
    assert paylar.between(0.999, 1.001).all(), "kanal payları 1'e toplanmıyor"


def test_top_category_gercekten_en_buyuk_pay(tablo):
    """Agent'ın Türkçe gerekçesi bu alandan çıkıyor; yanlışsa metin de yanlış olur."""
    pay_kolonlari = [f"cat_share_{c}" for c in CATEGORIES]
    beklenen = tablo[pay_kolonlari].idxmax(axis=1).str.replace("cat_share_", "", regex=False)
    assert (tablo["top_category"] == beklenen).all()
    assert (tablo["top_category_share"] - tablo[pay_kolonlari].max(axis=1)).abs().max() < 1e-9


def test_tabloda_eksik_deger_yok(tablo):
    """NaN, model eğitiminde sessizce satır düşürür — burada yakalanmalı."""
    assert tablo.select_dtypes(include="number").isna().sum().sum() == 0
    assert tablo["top_category"].notna().all()


def test_momentum_ve_oranlar_makul_aralikta(tablo):
    assert tablo["spend_momentum"].between(0, 10).all()
    assert tablo["installment_ratio"].between(0, 1).all()
    assert tablo["top_category_share"].between(0, 1).all()
    assert (tablo["recency_days"] >= 0).all()


# --------------------------------------------------------------------------
# Davranışsal özellik seti
# --------------------------------------------------------------------------


def test_davranissal_ozellikler_tabloda_var(tablo):
    eksik = [c for c in BEHAVIORAL_FEATURES if c not in tablo.columns]
    assert not eksik, f"BEHAVIORAL_FEATURES tabloda bulunamadı: {eksik}"


def test_davranissal_sette_demografi_yok(tablo):
    """Segmentasyon DAVRANIŞA bakar; yaş/gelir/ticari segment girerse bankanın
    zaten sahip olduğu bölümleme tekrar üretilmiş olur ve model gereksizleşir."""
    yasak = {"age", "tenure_months", "customer_segment", "income_band",
             "has_credit_card", "has_loan", "has_deposit", "digital_active"}
    assert not (set(BEHAVIORAL_FEATURES) & yasak)


def test_recency_kumelemeden_cikarilmis():
    """Bu veri setinde recency ayırt edici değil (ort ~2 gün); StandardScaler onu
    diğerleriyle eşit ağırlığa çıkarıp gürültü katardı. Model tarafında kalıyor."""
    assert "recency_days" not in BEHAVIORAL_FEATURES


# --------------------------------------------------------------------------
# Segmentasyon
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def segmentasyon(config_ve_tablo):
    config, tablo = config_ve_tablo
    sonuc, paket, profil = segment_et(tablo, config)
    return sonuc, paket, profil


def test_her_musteri_bir_segmente_atanmis(segmentasyon, tablo):
    sonuc, _, _ = segmentasyon
    assert len(sonuc) == len(tablo)
    assert sonuc["behavior_segment"].notna().all()


def test_mikro_kume_uretilmemis(segmentasyon):
    """Segment bir kampanya hedef kitlesidir; 20 kişilik küme iş olarak kullanılamaz."""
    sonuc, _, _ = segmentasyon
    oranlar = sonuc["behavior_cluster"].value_counts(normalize=True)
    assert oranlar.min() >= MIN_KUME_ORANI, f"çok küçük küme var: %{oranlar.min()*100:.1f}"


def test_segment_isimleri_benzersiz(segmentasyon):
    """İki farklı segment aynı etikete düşerse demoda ayırt edilemez."""
    _, paket, _ = segmentasyon
    isimler = list(paket["cluster_names"].values())
    assert len(isimler) == len(set(isimler))


def test_secilen_k_konfigdeki_aralikta(segmentasyon, config_ve_tablo):
    config, _ = config_ve_tablo
    _, paket, _ = segmentasyon
    assert paket["chosen_k"] in config["models"]["segmentation"]["k_range"]


def test_segmentler_davranissal_olarak_ayrisiyor(segmentasyon):
    """Kümeler gerçekten farklı harcama profilleri mi, yoksa rastgele bölme mi?

    En az bir kategoride, en yüksek payı olan küme en düşüğün iki katı olmalı.
    Bu sağlanmıyorsa kümeleme iş anlamı taşımıyordur.
    """
    _, _, profil = segmentasyon
    pay_kolonlari = [c for c in profil.columns if c.startswith("cat_share_")]
    oranlar = profil[pay_kolonlari].max() / profil[pay_kolonlari].min().replace(0, 1e-9)
    assert oranlar.max() >= 2.0, "kümeler arasında belirgin kategori farkı yok"
