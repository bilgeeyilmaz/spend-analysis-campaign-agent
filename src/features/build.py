"""Özellik tablosunu kurar ve diske yazar.

    python -m src.features.build

Çıktı: `data/processed/customer_features.parquet` — müşteri başına tek satır.

Bu modül, veri katmanı ile model katmanı arasındaki tek köprüdür. Veri kaynağına
`build_data_source()` üzerinden erişir; JSON mı SQL mi olduğunu bilmez.

DİKKAT — özellik seçimi burada bir MÜHENDİSLİK kararı değil, aynı zamanda bir
UYUM kararıdır. `EXCLUDED_FEATURES` listesindeki alanlar tabloya hiç girmez;
gerekçeleri aşağıda tek tek yazılı. Bir alanı listeden çıkarmadan önce gerekçesini
çürütebildiğinden emin ol.
"""

from __future__ import annotations

import argparse

import sys

import pandas as pd

from src.config import load_config, resolve_path
from src.data.source import build_data_source
from src.features.categories import (
    CATEGORIES,
    CHANNELS,
    category_features,
    channel_and_installment_features,
)
from src.features.rfm import compute_rfm

#: Modele ASLA girmeyecek alanlar ve gerekçeleri.
#:
#: * `gender`  — cinsiyete göre farklılaşan teklif/fiyat hukuki risk taşır.
#:               Şemada veri bütünlüğü için duruyor, buraya taşınmıyor.
#: * `city`    — 81 seviyeli, model için gürültü; demoda coğrafi ayrımcılık
#:               tartışması açar, karşılığında bir şey kazandırmaz.
EXCLUDED_FEATURES: set[str] = {"gender", "city"}

#: Tabloda DURAN ama modele girmeyen yasal kapı(lar).
#:
#: `opt_in_marketing` bir sinyal değil, bir izin. Modele girerse "izinsiz müşteri
#: zaten kabul etmiyor" korelasyonu öğrenilir ve kapının kendisi bir skora dönüşür;
#: oysa izin kontrolü `check_eligibility` içinde, modelden ÖNCE ve modelin
#: çıktısından bağımsız yapılmak zorunda. Tabloda tutuluyor çünkü uygunluk
#: kontrolü ve raporlama okuyor.
LEGAL_GATE_COLUMNS: set[str] = {"opt_in_marketing"}

#: Tabloda duran ama model girdisi olmayan alanlar (kimlik, metin, uygunluk girdisi).
NON_FEATURE_COLUMNS: set[str] = {
    "customer_id",
    "top_category",        # string; agent metni ve grafikler için
    "created_at",
}

#: Davranışsal segmentasyonun (Gün 4, KMeans) kullanacağı alanlar.
#:
#: Demografi ve bankanın ticari segmenti BİLİNÇLİ olarak dışarıda: amaç
#: "affluent müşteriler" gibi zaten bilinen bir bölümlemeyi tekrar üretmek değil,
#: HARCAMA DAVRANIŞINA göre yeni bir bölümleme çıkarmak. Sunumdaki "zaten
#: segmentiniz var, niye model kuruyorsunuz?" sorusunun cevabı bu ayrımdır.
#: `recency_days` BİLİNÇLİ olarak yok. EDA'da ortalaması 1,8 gün, %90'ı 5 günün
#: altında çıktı — bu veri setinde herkes aktif, yani ayırt edici bilgi taşımıyor.
#: StandardScaler düşük varyanslı bir kolonu diğerleriyle eşit ağırlığa çıkardığı
#: için kümelemeye gürültü olarak girerdi. Modelde (Gün 5) kalıyor, sadece
#: kümelemeden çıkarıldı. Gerçek banka verisinde recency muhtemelen geri gelir.
BEHAVIORAL_FEATURES: list[str] = (
    [f"cat_share_{c}" for c in CATEGORIES]
    + [f"channel_share_{c}" for c in CHANNELS]
    + [
        "log_monetary_total",
        "log_frequency_total",
        "spend_momentum",
        "installment_ratio",
        "avg_ticket",
        "active_months",
    ]
)


def resolve_as_of(config: dict, transactions: pd.DataFrame) -> pd.Timestamp:
    """Özelliklerin hesaplanacağı referans tarih.

    `config.features.as_of_date` boşsa verideki en son işlem tarihi kullanılır.
    Sabit vermek, geçmişe dönük yeniden üretimi (aynı girdiden aynı tablo)
    mümkün kılar; gerçek veriye geçince veri kesim tarihi buraya yazılır.
    """
    ayar = config["features"].get("as_of_date")
    return pd.Timestamp(ayar) if ayar else transactions["transaction_date"].max()


def build_feature_table(config: dict | None = None) -> pd.DataFrame:
    """Müşteri başına tek satırlık özellik tablosunu üretir."""
    config = config or load_config()
    source = build_data_source(config)

    customers = source.get_customers()
    transactions = source.get_transactions()
    if transactions.empty:
        raise ValueError(
            "İşlem tablosu boş. Önce sentetik veriyi üret: python -m src.data.generator"
        )

    as_of = resolve_as_of(config, transactions)
    recent = config["features"]["recent_window_days"]
    long = config["features"]["long_window_days"]

    parcalar = [
        compute_rfm(transactions, as_of, recent, long),
        category_features(transactions, as_of, recent, long),
        channel_and_installment_features(transactions),
    ]

    tablo = customers.drop(columns=[c for c in EXCLUDED_FEATURES if c in customers.columns])

    for parca in parcalar:
        tablo = tablo.merge(parca, on="customer_id", how="left")

    # Hiç işlemi olmayan müşteri. Sırası önemli: recency önce doldurulmalı, yoksa
    # genel fillna(0) onu "bugün işlem yaptı" anlamına gelen 0'a çeker — hiç
    # işlem yapmamış müşteri için tam ters sinyal.
    en_eski = (as_of - transactions["transaction_date"].min()).days
    tablo["recency_days"] = tablo["recency_days"].fillna(en_eski)
    tablo["avg_installment"] = tablo["avg_installment"].fillna(1.0)
    tablo["top_category"] = tablo["top_category"].fillna("diger")

    sayisal = tablo.select_dtypes(include="number").columns
    tablo[sayisal] = tablo[sayisal].fillna(0.0)

    tablo.attrs["as_of"] = str(as_of)
    return tablo


def model_feature_columns(tablo: pd.DataFrame) -> list[str]:
    """Propensity modeline verilebilecek kolonlar (Gün 5 bunu kullanacak).

    Dışlananlar tek tek `EXCLUDED_FEATURES` ve `NON_FEATURE_COLUMNS` içinde
    gerekçelendirilmiştir; burada ayrıca `opt_in_marketing` de çıkarılır çünkü
    tabloda bilinçli olarak duruyor ama modele girmiyor.
    """
    disarida = EXCLUDED_FEATURES | NON_FEATURE_COLUMNS | LEGAL_GATE_COLUMNS
    return [c for c in tablo.columns if c not in disarida]


def main(argv: list[str] | None = None) -> int:
    # Argparse yalnız --help için: bu komut argüman almıyor ama parser olmadan
    # `python -m src.features.build --help` yardım basmak yerine tabloyu
    # yeniden üretip diske yazıyordu.
    argparse.ArgumentParser(
        description="Müşteri özellik tablosunu üretir -> customer_features.parquet"
    ).parse_args(argv)

    config = load_config()
    tablo = build_feature_table(config)

    out_dir = resolve_path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "customer_features.parquet"
    tablo.to_parquet(path, index=False)

    sayisal = tablo.select_dtypes(include="number")
    print("\n=== Özellik Tablosu ===\n")
    print(f"  satır (müşteri)      : {len(tablo):,}")
    print(f"  kolon                : {len(tablo.columns)}")
    print(f"  model girdisi        : {len(model_feature_columns(tablo))}")
    print(f"  davranışsal (KMeans) : {len(BEHAVIORAL_FEATURES)}")
    print(f"  referans tarih       : {tablo.attrs['as_of'][:10]}")

    print("\n--- Modele girmeyen alanlar (gerekçeli) ---")
    for alan in sorted(EXCLUDED_FEATURES):
        print(f"  {alan:<20} tabloya hiç alınmadı")
    for alan in sorted(LEGAL_GATE_COLUMNS):
        print(f"  {alan:<20} tabloda var, model girdisi değil (yasal kapı)")

    print("\n--- Seçilmiş metrikler ---")
    for kolon in ("monetary_total", "monetary_30d", "frequency_total",
                  "recency_days", "spend_momentum", "installment_ratio",
                  "top_category_share", "distinct_categories"):
        s = sayisal[kolon]
        print(f"  {kolon:<20} ort {s.mean():>10,.2f}  medyan {s.median():>10,.2f}  "
              f"min {s.min():>9,.2f}  max {s.max():>12,.2f}")

    print("\n--- En sık baskın kategori ---")
    for kategori, adet in tablo["top_category"].value_counts().head(6).items():
        print(f"  {kategori:<18} {adet:>5,} müşteri ({adet/len(tablo):>5.1%})")

    bos = sayisal.isna().sum().sum()
    print(f"\n  eksik (NaN) hücre    : {bos}")
    print(f"  yazıldı              : {path} ({path.stat().st_size/1e6:.1f} MB)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
