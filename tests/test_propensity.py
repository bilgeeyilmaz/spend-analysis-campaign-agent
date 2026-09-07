"""Propensity modeli testleri.

Buradaki testlerin çoğu doğruluk değil, **ölçümün dürüstlüğü** testidir. Model
metriği sessizce yükselebilecek üç yol var ve üçü de sahada karşılığı olmayan
yollar:

  1. `responded_at` gibi etiket sonrası bir alanın özelliklere sızması,
  2. bir müşterinin tekliflerinin hem eğitimde hem testte bulunması,
  3. `campaign_id`'nin özellik olması — o an AUC'yi yükseltir ama katalogdaki
     her yeni kampanya için yeniden eğitim gerektirir.

Bu testler asıl olarak üçünü kapatıyor.

Testler kendi verisini üretir; `data/raw/` ve `models/` altındaki gerçek
çıktılara dokunmaz.
"""

from __future__ import annotations

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data import generator
from src.data.source import build_data_source
from src.features.build import build_feature_table
from src.models.propensity import (
    CATEGORICAL_FEATURES,
    LEAK_COLUMNS,
    PAIR_COLUMNS,
    add_cross_features,
    build_pair_frame,
    build_training_data,
    campaign_feature_frame,
    feature_columns,
    precision_at_1,
    score_campaigns,
    split_by_customer,
    train,
)

N_CUSTOMERS = 220


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> dict:
    """Küçük bir veri seti üretip özellik tablosunu diske yazar."""
    raw_dir = tmp_path_factory.mktemp("raw")
    processed_dir = tmp_path_factory.mktemp("processed")
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "campaigns.json", raw_dir / "campaigns.json")
    assert generator.main([
        "--n-customers", str(N_CUSTOMERS), "--out", str(raw_dir),
        "--as-of", "2026-08-13", "--seed", "7",
    ]) == 0

    cfg = copy.deepcopy(load_config())
    cfg["data"]["json"]["raw_dir"] = str(raw_dir)
    cfg["data"]["processed_dir"] = str(processed_dir)
    build_feature_table(cfg).to_parquet(processed_dir / "customer_features.parquet", index=False)

    # Test süresini makul tutmak için ızgarayı küçült; seçim mantığı aynı.
    cfg["models"]["propensity"]["param_grid"] = {
        "learning_rate": [0.05], "num_leaves": [4, 16], "n_estimators": [100],
    }
    cfg["models"]["propensity"]["algorithm"] = "hist"
    return cfg


@pytest.fixture(scope="module")
def egitim(config):
    return build_training_data(config)


@pytest.fixture(scope="module")
def paket(config):
    paket, _ = train(config, algorithm="hist")
    return paket


# --------------------------------------------------------------------------
# Sızıntı: etiket sonrası alanlar
# --------------------------------------------------------------------------


def test_responded_at_egitim_cercevesine_girmez(egitim):
    """Kabul eden her müşteride dolu olan alan; etiketin kendisidir."""
    df, X, _, _ = egitim
    assert "responded_at" in LEAK_COLUMNS
    for kolon in LEAK_COLUMNS:
        assert kolon not in df.columns, f"{kolon} eğitim çerçevesine sızmış"
        assert kolon not in X.columns


def test_interaction_tablosundan_sadece_uc_kolon_alinir(config):
    source = build_data_source(config)
    interactions = source.get_interactions()
    assert "responded_at" in interactions.columns, "üreteç değişmiş; test güncellenmeli"
    assert set(PAIR_COLUMNS) == {"customer_id", "campaign_id", "accepted"}


def test_sizintili_kolonla_cerceve_kurulmaz(config):
    """`build_pair_frame` yasak kolonu sessizce düşürmez, hata verir."""
    source = build_data_source(config)
    pairs = source.get_interactions()[PAIR_COLUMNS + ["responded_at"]]
    features = pd.read_parquet(
        f"{config['data']['processed_dir']}/customer_features.parquet"
    )
    with pytest.raises(ValueError, match="sızıntı"):
        build_pair_frame(pairs, features, source.get_campaigns())


# --------------------------------------------------------------------------
# Sızıntı: bölme
# --------------------------------------------------------------------------


def test_bolme_musteri_bazinda_ayrik(egitim):
    """Bir müşterinin teklifleri ya tamamen eğitimde ya tamamen testte olmalı.

    Aksi hâlde model kişiyi tanıyıp gizli eğilimini ezberler; ölçtüğümüz AUC
    yeni müşteride göreceğimizden yüksek çıkar.
    """
    _, _, _, groups = egitim
    tr, te = split_by_customer(groups, test_size=0.2, random_state=42)
    assert not set(groups[tr]) & set(groups[te])
    assert len(tr) + len(te) == len(groups)


def test_musteri_satirlarinda_musteri_kolonlari_sabit(egitim):
    """Bölmenin neden müşteri bazlı olması gerektiğinin veri üzerindeki kanıtı."""
    df, X, _, groups = egitim
    coklu = pd.Series(groups).value_counts()
    cid = coklu[coklu > 1].index[0]
    satirlar = X[groups == cid]
    sabit = satirlar["monetary_total"].nunique() == 1 and satirlar["age"].nunique() == 1
    assert sabit, "müşteri özellikleri satırdan satıra değişiyor — varsayım bozulmuş"


# --------------------------------------------------------------------------
# Tek model, tüm kampanyalar
# --------------------------------------------------------------------------


def test_campaign_id_ozellik_degil(egitim):
    """Kampanya kimliği özellik olsaydı yeni kampanya için yeniden eğitim gerekirdi."""
    _, X, _, _ = egitim
    assert "campaign_id" not in X.columns
    assert not any(k.startswith("campaign_id") for k in X.columns)


def test_yasak_kolonlar_feature_columns_tarafindan_reddedilir(config, egitim):
    df, _, _, _ = egitim
    features = pd.read_parquet(
        f"{config['data']['processed_dir']}/customer_features.parquet"
    )
    kolonlar = feature_columns(df, features)
    for yasak in ("campaign_id", "customer_id", "accepted", "offered_at", "responded_at"):
        assert yasak not in kolonlar


def test_uyum_ve_hukuki_alanlar_modelde_yok(config, egitim):
    """`opt_in_marketing`, `gender`, `city` propensity modeline de girmez."""
    _, X, _, _ = egitim
    for alan in ("opt_in_marketing", "gender", "city"):
        assert alan not in X.columns


# --------------------------------------------------------------------------
# Çapraz özellikler
# --------------------------------------------------------------------------


def test_capraz_ozellikler_hedef_kategorilerden_hesaplanir():
    """`fit_category_share`, kampanyanın hedeflediği kategorilerin payı toplamıdır."""
    df = pd.DataFrame([{
        "target_categories": ["market", "seyahat"],
        "min_spend": 1000.0, "min_tenure_months": 6, "min_age": 18,
        "offer_channel": "push", "digital_active": True,
        "tenure_months": 24, "age": 40,
        **{f"cat_share_{k}": 0.0 for k in _kategoriler()},
        **{f"cat_monthly_{k}": 0.0 for k in _kategoriler()},
        **{f"cat_trend_{k}": 0.0 for k in _kategoriler()},
    }])
    df.loc[0, "cat_share_market"] = 0.30
    df.loc[0, "cat_share_seyahat"] = 0.12
    df.loc[0, "cat_share_giyim"] = 0.58          # hedefte değil, sayılmamalı
    df.loc[0, "cat_monthly_market"] = 500.0

    out = add_cross_features(df)
    assert out.loc[0, "fit_category_share"] == pytest.approx(0.42)
    assert out.loc[0, "fit_monthly_amount"] == pytest.approx(500.0)
    assert out.loc[0, "min_spend_friction"] == pytest.approx(2.0)
    assert out.loc[0, "channel_match"] == 1      # dijital müşteri, push kanalı
    assert out.loc[0, "tenure_margin"] == 18
    assert "target_categories" not in out.columns


def test_channel_match_dijital_olmayan_musteride_sms():
    ortak = {
        "target_categories": [], "min_spend": 100.0, "min_tenure_months": 0,
        "min_age": 18, "tenure_months": 12, "age": 30,
        **{f"cat_share_{k}": 0.0 for k in _kategoriler()},
        **{f"cat_monthly_{k}": 0.0 for k in _kategoriler()},
        **{f"cat_trend_{k}": 0.0 for k in _kategoriler()},
    }
    df = pd.DataFrame([
        {**ortak, "offer_channel": "sms", "digital_active": False},
        {**ortak, "offer_channel": "push", "digital_active": False},
    ])
    out = add_cross_features(df)
    assert out.loc[0, "channel_match"] == 1
    assert out.loc[1, "channel_match"] == 0


def test_kampanya_bloğu_min_age_ozellik_olarak_kalmaz(config):
    """`min_age` çapraz blok için taşınır, özellik olarak kalmaz."""
    source = build_data_source(config)
    camp = campaign_feature_frame(source.get_campaigns())
    assert "min_age" in camp.columns and "target_categories" in camp.columns
    _, X, _, _ = build_training_data(config)
    assert "min_age" not in X.columns
    assert "target_categories" not in X.columns


def _kategoriler() -> list[str]:
    from src.features.categories import CATEGORIES
    return CATEGORIES


# --------------------------------------------------------------------------
# Metrikler ve eğitim
# --------------------------------------------------------------------------


def test_precision_at_1_tek_teklifli_musteriyi_saymaz():
    musteri = np.array(["A", "A", "B"])
    y = np.array([0, 1, 1])
    skor = np.array([0.1, 0.9, 0.5])
    oran, n = precision_at_1(musteri, y, skor, min_offers=2)
    assert n == 1                 # B tek teklif almış, elendi
    assert oran == pytest.approx(1.0)


def test_egitim_metrikleri_makul_bantta(paket):
    """AUC bandı: rastgeleden belirgin yukarıda, tavanı fahiş aşmayan.

    Üst sınır gevşek çünkü test 220 müşteriyle koşuyor; asıl kalibrasyon
    `python -m src.models.propensity` çıktısında görülür.
    """
    m = paket["metrics"]
    assert 0.60 < m["auc"] < 0.95, m["auc"]
    assert m["pr_auc"] > paket["acceptance_rate"]
    assert m["precision_at_1"] > paket["acceptance_rate"]


def test_model_rastgele_baselini_geciyor(paket):
    assert paket["metrics"]["auc"] > paket["baselines"]["rastgele"]["auc"] + 0.10
    assert paket["metrics"]["uplift_vs_random"] > 1.2


def test_paket_skorlama_icin_gereken_her_seyi_tasiyor(paket):
    for anahtar in ("model", "feature_columns", "categories", "params", "metrics"):
        assert anahtar in paket
    for kolon in paket["categorical_features"]:
        assert kolon in CATEGORICAL_FEATURES
        assert paket["categories"][kolon], f"{kolon} kategori listesi boş"


def test_sizinti_kontrolu_raporlaniyor(paket):
    lk = paket["leakage_check"]
    assert lk["shared_customers"] > 0, "rastgele bölmede ortak müşteri çıkmadı"
    assert lk["grouped_auc"] == paket["metrics"]["auc"]


# --------------------------------------------------------------------------
# Skorlama yolu (Gün 6+ araç katmanı bunu çağıracak)
# --------------------------------------------------------------------------


def test_score_campaigns_sirali_skor_dondurur(config, paket):
    source = build_data_source(config)
    features = pd.read_parquet(
        f"{config['data']['processed_dir']}/customer_features.parquet"
    )
    musteri = features.head(1)
    campaigns = source.get_campaigns()

    sonuc = score_campaigns(paket, musteri, campaigns)
    assert len(sonuc) == len(campaigns)
    assert sonuc["score"].between(0, 1).all()
    assert sonuc["score"].is_monotonic_decreasing
    assert set(sonuc["campaign_id"]) == set(campaigns["campaign_id"])


def test_score_campaigns_bos_katalogda_bos_doner(config, paket):
    features = pd.read_parquet(
        f"{config['data']['processed_dir']}/customer_features.parquet"
    )
    bos = build_data_source(config).get_campaigns().head(0)
    assert score_campaigns(paket, features.head(1), bos).empty


def test_score_campaigns_tek_musteri_bekler(config, paket):
    features = pd.read_parquet(
        f"{config['data']['processed_dir']}/customer_features.parquet"
    )
    campaigns = build_data_source(config).get_campaigns()
    with pytest.raises(ValueError, match="tek müşteri"):
        score_campaigns(paket, features.head(2), campaigns)
