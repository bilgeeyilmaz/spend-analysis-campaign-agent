"""Veri sözleşmesi testleri.

Bu testler projenin ilk güvenlik ağıdır: şema bozulursa aşağıdaki tüm
katmanlar sessizce yanlış çalışacağı için hatayı burada yakalamak istiyoruz.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT
from src.data.schemas import (
    Campaign,
    Customer,
    CustomerSegment,
    IncomeBand,
    Interaction,
    MccCategory,
    OfferChannel,
    TABLE_SCHEMAS,
    required_columns,
)

CAMPAIGN_CATALOG = PROJECT_ROOT / "data" / "raw" / "campaigns.json"


# --------------------------------------------------------------------------
# Kampanya kataloğu — elle yazıldığı için makine tarafından doğrulanmalı
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    with CAMPAIGN_CATALOG.open(encoding="utf-8") as f:
        return json.load(f)


def test_katalog_semaya_uyuyor(catalog):
    """Katalogdaki her kampanya Campaign şemasından geçmeli."""
    for raw in catalog:
        Campaign.model_validate(raw)


def test_kampanya_idleri_benzersiz(catalog):
    ids = [c["campaign_id"] for c in catalog]
    assert len(ids) == len(set(ids)), "Katalogda tekrar eden campaign_id var"


def test_katalog_yeterince_cesitli(catalog):
    """Demo ve model için kategori/ödül çeşitliliği şart.

    Tek kategoriye yığılmış katalogda öneri motoru anlamsızlaşır.
    """
    campaigns = [Campaign.model_validate(c) for c in catalog]

    kategoriler = {cat for c in campaigns for cat in c.target_categories}
    assert len(kategoriler) >= 8, f"Sadece {len(kategoriler)} kategori hedefleniyor"

    odul_tipleri = {c.reward_type for c in campaigns}
    assert len(odul_tipleri) >= 3, "Ödül tipi çeşitliliği yetersiz"


def test_katalogta_aktif_ve_pasif_kampanya_var(catalog):
    """active_only filtresinin gerçekten bir iş yaptığını garanti eder."""
    campaigns = [Campaign.model_validate(c) for c in catalog]
    as_of = date(2026, 8, 10)

    aktif = [c for c in campaigns if c.is_active(as_of)]
    pasif = [c for c in campaigns if not c.is_active(as_of)]

    assert len(aktif) >= 8, "Demo için yeterli aktif kampanya yok"
    assert pasif, "Filtreyi test edecek pasif kampanya yok"


def test_suresi_dolmus_kampanya_aktif_sayilmaz(catalog):
    campaigns = {c["campaign_id"]: Campaign.model_validate(c) for c in catalog}
    assert not campaigns["KMP013"].is_active(date(2026, 8, 10)), "Tarihi geçmiş kampanya aktif görünüyor"
    assert not campaigns["KMP014"].is_active(date(2026, 8, 10)), "Bütçesi biten kampanya aktif görünüyor"


# --------------------------------------------------------------------------
# Şema kuralları gerçekten zorlanıyor mu
# --------------------------------------------------------------------------


def _gecerli_musteri(**overrides) -> dict:
    payload = {
        "customer_id": "C000001",
        "age": 34,
        "gender": "K",
        "city": "Adana",
        "customer_segment": CustomerSegment.MASS,
        "income_band": IncomeBand.C,
        "tenure_months": 48,
        "has_credit_card": True,
        "has_debit_card": True,
        "has_loan": False,
        "has_deposit": True,
        "digital_active": True,
        "opt_in_marketing": True,
        "created_at": date(2022, 8, 1),
    }
    payload.update(overrides)
    return payload


def test_gecerli_musteri_kabul_edilir():
    Customer.model_validate(_gecerli_musteri())


@pytest.mark.parametrize(
    "alan, deger",
    [
        ("age", 15),          # 18 yaş altı
        ("age", 130),         # üst sınır
        ("tenure_months", -1),
        ("customer_id", "  "),
    ],
)
def test_gecersiz_musteri_reddedilir(alan, deger):
    with pytest.raises(ValidationError):
        Customer.model_validate(_gecerli_musteri(**{alan: deger}))


def _gecerli_kampanya(**overrides) -> dict:
    payload = {
        "campaign_id": "TEST01",
        "name": "Test",
        "description": "Test kampanyası",
        "target_categories": [MccCategory.MARKET],
        "reward_type": "cashback",
        "reward_value": 10.0,
        "reward_unit": "yuzde",
        "min_spend": 500.0,
        "valid_from": date(2026, 8, 1),
        "valid_to": date(2026, 9, 1),
        "total_budget": 100000.0,
        "remaining_budget": 50000.0,
        "offer_channel": OfferChannel.SMS,
    }
    payload.update(overrides)
    return payload


def test_bitis_tarihi_baslangictan_once_olamaz():
    with pytest.raises(ValidationError, match="valid_to"):
        Campaign.model_validate(
            _gecerli_kampanya(valid_from=date(2026, 9, 1), valid_to=date(2026, 8, 1))
        )


def test_kalan_butce_toplami_asamaz():
    with pytest.raises(ValidationError, match="remaining_budget"):
        Campaign.model_validate(
            _gecerli_kampanya(total_budget=1000.0, remaining_budget=5000.0)
        )


def test_kampanya_en_az_bir_kategori_hedeflemeli():
    with pytest.raises(ValidationError):
        Campaign.model_validate(_gecerli_kampanya(target_categories=[]))


def test_cevap_tarihi_tekliften_once_olamaz():
    with pytest.raises(ValidationError, match="responded_at"):
        Interaction.model_validate(
            {
                "interaction_id": "I1",
                "customer_id": "C000001",
                "campaign_id": "KMP001",
                "offered_at": datetime(2026, 8, 10, 12, 0),
                "offer_channel": OfferChannel.SMS,
                "accepted": True,
                "responded_at": datetime(2026, 8, 9, 12, 0),
            }
        )


# --------------------------------------------------------------------------
# PII politikası — şema seviyesinde garanti
# --------------------------------------------------------------------------


def test_musteri_semasinda_pii_alani_yok():
    """Şemada PII olmaması, LLM'e PII gitmesini kökten imkânsız kılar."""
    yasakli = {
        "name", "first_name", "last_name", "ad", "soyad", "isim",
        "tckn", "national_id", "phone", "telefon", "gsm",
        "email", "eposta", "iban", "card_number", "kart_no", "address", "adres",
    }
    alanlar = set(Customer.model_fields.keys())
    assert not (alanlar & yasakli), f"Customer şemasında PII alanı var: {alanlar & yasakli}"


# --------------------------------------------------------------------------
# Sözleşme kaydı
# --------------------------------------------------------------------------


def test_required_columns_tum_tablolar_icin_calisiyor():
    for tablo in TABLE_SCHEMAS:
        kolonlar = required_columns(tablo)
        assert kolonlar, f"{tablo} için kolon listesi boş"
        assert "customer_id" in kolonlar or tablo == "campaigns"


def test_bilinmeyen_tablo_hata_verir():
    with pytest.raises(KeyError):
        required_columns("olmayan_tablo")
