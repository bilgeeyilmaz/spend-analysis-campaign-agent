"""Sentetik veri üreteci testleri.

Üreteç projenin temelidir: buradaki bir hata sessizce özelliklere, modele ve
agent'a taşınır. Bu yüzden testler üç şeyi ayrı ayrı zorluyor:

  1. **Sözleşme** — üretilen kayıtlar `schemas.py`'den ve `validate_source()`'tan geçiyor mu
  2. **Yasal/etik kısıtlar** — PII yok, gizli persona sızmıyor, izinsiz müşteriye teklif yok
  3. **PFM sinyalleri** — `analyze_spending`'in bulacağı düzenli gider ve sıçrama
     gerçekten ekilmiş mi (ekilmezse Gün 6'daki araç boş çıktı verir)

Testler küçük bir örneklemle (150 müşteri) çalışır; amaç istatistiksel kesinlik
değil, yapının bozulmadığını doğrulamaktır.
"""

from __future__ import annotations

import json
import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data import generator
from src.data.schemas import Customer, Interaction, Transaction
from src.data.source import JsonDataSource
from src.data.validate import validate_source

N_CUSTOMERS = 150
AS_OF = date(2026, 8, 13)


@pytest.fixture(scope="module")
def uretilmis_veri(tmp_path_factory) -> dict:
    """Üreteci gerçek CLI yolundan çalıştırır ve çıktıyı okur.

    Geçici dizine yazar: testler `data/raw/` altındaki gerçek veriyi ezmemeli.
    `campaigns.json` üretilmediği için kopyalanır — üretecin katalogu okuyup
    yazmadığı da böylece test edilmiş olur.
    """
    out_dir = tmp_path_factory.mktemp("raw")
    katalog = PROJECT_ROOT / "data" / "raw" / "campaigns.json"
    shutil.copy(katalog, out_dir / "campaigns.json")
    katalog_ozgun = katalog.read_bytes()

    exit_code = generator.main([
        "--n-customers", str(N_CUSTOMERS),
        "--out", str(out_dir),
        "--as-of", AS_OF.isoformat(),
        "--seed", "42",
    ])
    assert exit_code == 0, "üreteç sıfırdan farklı çıkış kodu döndürdü"

    def _oku(ad: str) -> list[dict]:
        with (out_dir / ad).open(encoding="utf-8") as f:
            return json.load(f)

    return {
        "dir": out_dir,
        "customers": _oku("customers.json"),
        "transactions": _oku("transactions.json"),
        "interactions": _oku("interactions.json"),
        "katalog_ozgun": katalog_ozgun,
    }


# --------------------------------------------------------------------------
# 1) Sözleşme
# --------------------------------------------------------------------------


def test_tum_kayitlar_semadan_geciyor(uretilmis_veri):
    """Örnekleme değil, TÜM kayıtlar — üreteç hatası hiçbir satırda olmamalı."""
    for row in uretilmis_veri["customers"]:
        Customer.model_validate(row)
    for row in uretilmis_veri["transactions"]:
        Transaction.model_validate(row)
    for row in uretilmis_veri["interactions"]:
        Interaction.model_validate(row)


def test_validate_source_basarili(uretilmis_veri):
    """Üretilen veri, gerçek veri kaynağı doğrulayıcısından geçmeli.

    Bu test şema + referans bütünlüğü + PII + kabul oranı kontrollerini birden
    kapsar; `python -m src.data.validate` komutunun CI karşılığıdır.
    """
    report = validate_source(JsonDataSource(uretilmis_veri["dir"]))
    assert report.ok, f"doğrulama başarısız: {report.errors}"


def test_kimlikler_benzersiz(uretilmis_veri):
    for tablo, alan in (
        ("customers", "customer_id"),
        ("transactions", "transaction_id"),
        ("interactions", "interaction_id"),
    ):
        ids = [row[alan] for row in uretilmis_veri[tablo]]
        assert len(ids) == len(set(ids)), f"{tablo}: tekrar eden {alan}"


def test_katalog_uretecte_degistirilmedi(uretilmis_veri):
    """campaigns.json elle yazıldı ve versiyonlanıyor; üreteç ona dokunmamalı."""
    yazilmis = (uretilmis_veri["dir"] / "campaigns.json").read_bytes()
    assert yazilmis == uretilmis_veri["katalog_ozgun"]


def test_islemler_gecmis_penceresinde(uretilmis_veri):
    tarihler = pd.to_datetime([r["transaction_date"] for r in uretilmis_veri["transactions"]])
    n_months = load_config()["generator"]["n_months"]
    baslangic = generator._month_windows(AS_OF, n_months)[0][0]
    assert tarihler.max().date() <= AS_OF, "gelecek tarihli işlem var"
    assert tarihler.min().date() >= baslangic, "pencere dışında eski işlem var"


# --------------------------------------------------------------------------
# 2) Yasal / etik kısıtlar
# --------------------------------------------------------------------------


def test_izinsiz_musteriye_teklif_sunulmamis(uretilmis_veri):
    """KVKK: `opt_in_marketing=False` müşteriye geçmişte de teklif gitmemiş olmalı.

    Aksi halde veri, projenin en sert iddiasıyla (opt_in ilk kontroldür) çelişir.
    """
    izinsiz = {
        c["customer_id"] for c in uretilmis_veri["customers"] if not c["opt_in_marketing"]
    }
    teklif_alan = {i["customer_id"] for i in uretilmis_veri["interactions"]}
    assert not (izinsiz & teklif_alan), "izin vermemiş müşteriye teklif kaydı var"


def test_gizli_persona_veriye_sizmamis(uretilmis_veri):
    """Persona sızarsa Gün 4'teki KMeans sonucu anlamsızlaşır (cevabı görmüş olur)."""
    persona_adlari = set(generator.PERSONAS)
    for row in uretilmis_veri["customers"]:
        assert not (set(row) & {"persona", "propensity", "latent", "activity"})
        assert not (set(map(str, row.values())) & persona_adlari)


def test_uretilen_musteride_pii_alani_yok(uretilmis_veri):
    """Şema zaten engelliyor; bu test üretecin fazladan alan eklemediğini garanti eder."""
    izinli = set(Customer.model_fields.keys())
    for row in uretilmis_veri["customers"]:
        assert set(row) == izinli, f"beklenmeyen alan: {set(row) - izinli}"


# --------------------------------------------------------------------------
# 3) PFM sinyalleri — analyze_spending bunları bulacak
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def islem_df(uretilmis_veri) -> pd.DataFrame:
    df = pd.DataFrame(uretilmis_veri["transactions"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["ay"] = df["transaction_date"].dt.to_period("M")
    return df


def test_duzenli_odeme_tespit_edilebiliyor(islem_df):
    """Config eşikleriyle, üretim etiketine (merchant_id deseni) BAKMADAN tespit.

    Gün 6'da `analyze_spending` aynı kuralı uygulayacak; burada verinin o kuralı
    besleyip beslemediğini ölçüyoruz.
    """
    esik = load_config()["agent"]["spending_analysis"]
    alt, ust = esik["recurring_gap_days"]

    # (müşteri, üye işyeri, ay) -> aylık toplam. Müşteri başına döngü yerine tek
    # groupby: Gün 6'daki araç da bunu böyle yapmalı, 2.000 müşteride satır satır
    # dolaşmak kabul edilebilir bir gecikme vermez.
    aylik = (
        islem_df.groupby(["customer_id", "merchant_id", "ay"], observed=True)
        .agg(tutar=("amount", "sum"), adet=("amount", "size"),
             ilk=("transaction_date", "min"))
        .reset_index()
        .sort_values(["customer_id", "merchant_id", "ay"])
    )
    gun_farki = aylik.groupby(["customer_id", "merchant_id"], observed=True)["ilk"].diff().dt.days
    aylik["gap_ok"] = gun_farki.isna() | gun_farki.between(alt, ust)

    ozet = aylik.groupby(["customer_id", "merchant_id"], observed=True).agg(
        ay_sayisi=("tutar", "size"), tutar_min=("tutar", "min"),
        tutar_max=("tutar", "max"), tutar_ort=("tutar", "mean"),
        adet_max=("adet", "max"), gap_ok=("gap_ok", "all"),
    )
    oynama = (ozet["tutar_max"] - ozet["tutar_min"]) / ozet["tutar_ort"]

    duzenli = ozet[
        (ozet["ay_sayisi"] >= esik["recurring_min_months"])
        & (ozet["adet_max"] == 1)                    # ayda tek ödeme
        & (oynama <= esik["recurring_amount_tolerance"])
        & ozet["gap_ok"]
    ]
    oran = duzenli.reset_index()["customer_id"].nunique() / islem_df["customer_id"].nunique()
    assert oran > 0.40, f"düzenli gideri tespit edilen müşteri oranı çok düşük: {oran:.0%}"


def test_duzenli_odemesi_olmayan_musteri_de_var(islem_df):
    """Negatif fixture: `duzenli_giderler` boş dönen bir vaka olmalı.

    Herkeste düzenli ödeme olsaydı Gün 6'daki testin boş-liste dalı yazılamazdı.
    """
    duzenli_olan = set(islem_df[islem_df["merchant_id"].str.contains("_S")]["customer_id"])
    assert duzenli_olan, "hiç düzenli ödeme üretilmemiş"
    assert len(duzenli_olan) < islem_df["customer_id"].nunique(), \
        "her müşteride düzenli ödeme var — negatif fixture yok"


def test_olagandisi_artis_tespit_edilebiliyor(islem_df):
    """Son 3 ayda, config'teki dört şartı birden sağlayan sıçrama bulunmalı."""
    esik = load_config()["agent"]["spending_analysis"]
    son_aylar = sorted(islem_df["ay"].unique())[-3:]

    bulunan = 0
    for _, g in islem_df.groupby("customer_id"):
        pivot = g.pivot_table(index="ay", columns="mcc_category",
                              values="amount", aggfunc="sum").fillna(0.0)
        for ay in son_aylar:
            if ay not in pivot.index:
                continue
            onceki = pivot.loc[pivot.index < ay].tail(esik["anomaly_baseline_months"])
            if len(onceki) < esik["anomaly_baseline_months"]:
                continue
            for kat in pivot.columns:
                if esik["anomaly_require_full_baseline"] and (onceki[kat] == 0).any():
                    continue
                son, taban = pivot.loc[ay, kat], onceki[kat].mean()
                if (taban >= esik["anomaly_min_baseline"]
                        and son >= taban * esik["anomaly_ratio_threshold"]
                        and son - taban >= esik["anomaly_min_amount"]):
                    bulunan += 1
                    break
            else:
                continue
            break

    oran = bulunan / islem_df["customer_id"].nunique()
    assert oran > 0.05, f"hiç sıçrama tespit edilemiyor ({oran:.0%}) — araç boş çıktı verir"
    assert oran < 0.60, f"sıçrama oranı çok yüksek ({oran:.0%}) — uyarı anlamsızlaşır"


def test_kategori_dagilimi_config_hedefine_yakin(islem_df):
    """Kategori payları TCMB'ye kalibre edilmiş hedeften çok sapmamalı.

    `category_weights` TUTAR payıdır; üreteç bunu ortalama sepete bölerek adet
    olasılığına çevirir. Bu dönüşüm bozulursa (ilk sürümde bozuktu) seyahat/
    elektronik gibi yüksek sepetli kategoriler payı katlar.
    """
    hedef = load_config()["generator"]["category_weights"]
    gerceklesen = islem_df.groupby("mcc_category")["amount"].sum()
    gerceklesen /= gerceklesen.sum()

    for kategori, beklenen in hedef.items():
        assert kategori in gerceklesen.index, f"{kategori} hiç üretilmemiş"
        sapma = abs(gerceklesen[kategori] - beklenen)
        assert sapma < 0.06, (
            f"{kategori}: gerçekleşen {gerceklesen[kategori]:.1%}, hedef {beklenen:.1%}"
        )


# --------------------------------------------------------------------------
# 4) Etiket kalitesi — "döngüsellik tuzağı"
# --------------------------------------------------------------------------


def test_kabul_orani_makul_bantta(uretilmis_veri):
    accepted = [i["accepted"] for i in uretilmis_veri["interactions"]]
    oran = sum(accepted) / len(accepted)
    assert 0.05 <= oran <= 0.60, f"kabul oranı bant dışı: {oran:.1%}"
    assert 0 < sum(accepted) < len(accepted), "hedef değişken tek sınıftan oluşuyor"


def test_etiket_basit_bir_kuralla_uretilmemis(uretilmis_veri):
    """Kabul edilenlerde de reddedilenlerde de kampanya çeşitliliği olmalı.

    Etiket "şu kampanya hep kabul edilir" gibi bir kuralla üretilseydi kabul
    kümesi birkaç kampanyaya yığılırdı ve model AUC ~0.99 ile kuralı ezberlerdi.
    """
    kabul = {i["campaign_id"] for i in uretilmis_veri["interactions"] if i["accepted"]}
    ret = {i["campaign_id"] for i in uretilmis_veri["interactions"] if not i["accepted"]}
    assert len(kabul) >= 6, f"kabul edilen kampanya çeşitliliği düşük: {len(kabul)}"
    assert kabul & ret, "hiçbir kampanya hem kabul hem ret almamış — etiket determinist"


def test_cevap_tarihi_teklif_sonrasi(uretilmis_veri):
    for row in uretilmis_veri["interactions"]:
        if row["responded_at"] is not None:
            assert row["responded_at"] >= row["offered_at"]
