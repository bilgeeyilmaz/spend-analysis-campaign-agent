"""Denetçi testleri.

Testlerin çoğu **gerçekten gözlemlenmiş** LLM hatalarından türetildi; uydurma
senaryo değiller:

  * gpt-oss-20b: "aylık ortalama 9.215 TL harcayan bir giyim aboneliğiniz var"
    — veride böyle bir kayıt yok, tamamen uydurma.
  * gpt-oss-120b: "%25 nakit iade" — kampanya %8 veriyor; 25 müşterinin market
    payıydı. Sayı veride VAR ama yanlış yere bağlanmış.

İkisi farklı hata türü ve farklı denetim yakalıyor. Bir regresyon testi olarak
değerleri buraya sabitliyoruz.
"""

from __future__ import annotations

import pytest

from src.agent.guardrails import (
    arac_degerleri,
    denetle,
    izlenebilirlik_denetimi,
    odul_denetimi,
    sayilari_cikar,
    tavsiye_denetimi,
)

TOOL_CALLS = [
    {
        "name": "analyze_spending",
        "arguments": {"customer_id": "C000004"},
        "result": {
            "toplam_harcama": 53609.49,
            "islem_adedi": 45,
            "ortalama_sepet": 1191.32,
            "kategori_dagilimi": [
                {"kategori": "market", "pay": 0.2477, "tutar": 13279.0},
                {"kategori": "egitim", "pay": 0.2053},
            ],
            "olagandisi_artislar": [
                {"kategori": "giyim", "tutar": 6103.0, "kat": 7.6}
            ],
        },
    },
    {
        "name": "score_campaigns",
        "arguments": {"customer_id": "C000004"},
        "result": {"oneriler": [
            {"campaign_id": "KMP001", "name": "Market Alışverişine Nakit İade",
             "score": 0.5318}
        ]},
    },
    {
        "name": "get_campaign_details",
        "arguments": {"campaign_id": "KMP001"},
        "result": {"campaign_id": "KMP001", "reward_value": 8.0, "min_spend": 750.0},
    },
]

KATALOG = {"KMP001", "KMP008", "KMP014"}


# --------------------------------------------------------------------------
# Sayı ayrıştırma — Türkçe biçimler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metin,beklenen", [
    ("53.609,49 TL", 53609.49),      # noktalı binlik + virgüllü ondalık
    ("53 609 TL", 53609.0),          # boşluklu binlik (LLM bunu da üretiyor)
    ("53 609,49 TL", 53609.49),  # dar boşluk — LLM gerçekten bunu üretti
    ("53 609,49 TL", 53609.49),  # kırılmaz boşluk
    ("53 609,49 TL", 53609.49),  # ince boşluk
    ("7,6 katı", 7.6),
    ("7.6 katı", 7.6),               # nokta 3 rakam getirmiyorsa ONDALIKTIR
    ("1.191 TL", 1191.0),
    ("%25", 25.0),
])
def test_turkce_sayi_bicimleri(metin, beklenen):
    assert sayilari_cikar(metin)[0][0] == pytest.approx(beklenen)


def test_yuzde_isareti_algilanir():
    (_, yuzde_mi), = sayilari_cikar("%25")
    assert yuzde_mi is True
    (_, yuzde_mi2), = sayilari_cikar("25 işlem")
    assert yuzde_mi2 is False


def test_oran_yuzdeye_cevrilerek_eslesir():
    """Araç `pay: 0.2477` diyor, LLM "%25" yazıyor — bu meşru bir dönüşüm."""
    havuz = arac_degerleri(TOOL_CALLS)
    assert any(abs(d - 24.77) < 0.01 for d in havuz)


# --------------------------------------------------------------------------
# İzlenebilirlik: uydurma sayı
# --------------------------------------------------------------------------


def test_dogru_metin_denetimden_gecer():
    metin = (
        "Son 3 ayda 53.609 TL harcadınız. Harcamanızın %25'i market kategorisinde. "
        "Giyimde 6.103 TL'lik tek bir işlem var, alışılmışın 7,6 katı. "
        "\"Market Alışverişine Nakit İade\" (KMP001) kampanyası %8 nakit iade "
        "sağlıyor, asgari harcama 750 TL."
    )
    sonuc = denetle(metin, TOOL_CALLS, KATALOG)
    assert sonuc.gecti, sonuc.ihlaller


def test_uydurma_sayi_yakalanir():
    """gpt-oss-20b'nin gerçek hatası."""
    metin = "Düzenli olarak aylık ortalama 9.215 TL harcayan bir giyim aboneliğiniz var."
    assert 9215.0 in izlenebilirlik_denetimi(metin, TOOL_CALLS)


def test_yuvarlanmis_sayi_ihlal_sayilmaz():
    """LLM 53.609,49'u 53.609 ya da 54.000 diye yazabilir; bu uydurma değil."""
    assert izlenebilirlik_denetimi("Toplam 53.609 TL.", TOOL_CALLS) == []
    assert izlenebilirlik_denetimi("Yaklaşık 54.000 TL.", TOOL_CALLS) == []


def test_kucuk_baglam_sayilari_muaf():
    """"Son 3 ayda", "ilk 5 kategori" gibi ifadeler sürekli alarm üretmemeli."""
    assert izlenebilirlik_denetimi("Son 3 ayda ilk 5 kategoriye baktık.", TOOL_CALLS) == []


def test_metin_alanindaki_sayilar_da_havuza_girer():
    """Kampanya adı/açıklaması sayıyı metin içinde taşıyor.

    Bu gözden kaçınca denetçi, LLM rakamı DOĞRU alıntıladığında "uydurma"
    diyordu — yanlış alarmın ana kaynağı buydu.
    """
    tc = [{
        "name": "get_campaign_details", "arguments": {"campaign_id": "KMP010"},
        "result": {"campaign_id": "KMP010", "name": "Faturaya 50 TL İade",
                   "description": "Fatura talimatına 50 TL iade, asgari 200 TL."},
    }]
    assert izlenebilirlik_denetimi("50 TL iade, asgari 200 TL harcama.", tc) == []


def test_azalis_orani_yuzdeye_cevrilir():
    """`degisim_orani: -0.78` metinde "%78 azaldı" diye yazılıyor."""
    tc = [{"name": "analyze_spending", "arguments": {},
           "result": {"trend": {"degisim_orani": -0.78}}}]
    assert izlenebilirlik_denetimi("Harcamanız %78 azaldı.", tc) == []


def test_yuzde_yuzu_asan_artis_yuzdeye_cevrilir():
    """`degisim_orani: 1.0733` -> "%107". Oran 1'i aşabiliyor."""
    tc = [{"name": "analyze_spending", "arguments": {},
           "result": {"trend": {"degisim_orani": 1.0733}}}]
    assert izlenebilirlik_denetimi("Harcamanız %107 arttı.", tc) == []


@pytest.mark.parametrize("yazim", ["53.609", "53.610", "53.600", "54.000"])
def test_makul_yuvarlamalar_kabul_edilir(yazim):
    """Yuvarlama farkı toleransla değil, varyant üretilerek karşılanıyor.

    Toleransı gevşetmek tüm sayı eksenini genişletir ve uydurma tutarları da
    kapsardı; varyant yalnızca gerçek değerin makul yazımlarını ekler.
    """
    assert izlenebilirlik_denetimi(f"Toplam {yazim} TL.", TOOL_CALLS) == []


def test_dar_tolerans_uydurmayi_kacirmiyor():
    """Gerçek veride havuz yüzlerce sayı içeriyor; tolerans dar olmazsa
    izlenebilirlik denetimi anlamını yitiriyor (ölçüldü: %2 toleransta
    uydurulmuş tutarların yalnızca %70'i yakalanıyordu)."""
    assert izlenebilirlik_denetimi("Ayrıca 37.412 TL'lik bir ödemeniz var.", TOOL_CALLS)


def test_nan_denetciyi_cokertmez():
    """Boş seriden alınan ortalama NaN olabiliyor; `round(nan)` ValueError atar."""
    tc = [{"name": "analyze_spending", "arguments": {},
           "result": {"ortalama": float("nan"), "toplam": 100.0}}]
    assert izlenebilirlik_denetimi("Toplam 100 TL.", tc) == []


# --------------------------------------------------------------------------
# Atıf hatası: ödül oranı
# --------------------------------------------------------------------------


def test_yanlis_odul_orani_yakalanir():
    """gpt-oss-120b'nin gerçek hatası: %25 müşterinin market payı, ödül %8."""
    metin = "KMP001 kampanyası market harcamalarınıza %25 nakit iade sağlıyor."
    ihlaller = odul_denetimi(metin, TOOL_CALLS)
    assert ihlaller and "odul_uyusmazligi" in ihlaller[0]


def test_yanlis_odul_izlenebilirlikten_KACAR():
    """Bu testin varlık sebebi: iki denetimin neden ayrı olduğunu gösteriyor.

    25 sayısı araç çıktısında GERÇEKTEN var (market payı %24,77). Yani
    izlenebilirlik denetimi bu hatayı göremez — atıf hatasını ancak hedefli
    ödül kontrolü yakalar.
    """
    metin = "KMP001 kampanyası %25 nakit iade sağlıyor."
    assert izlenebilirlik_denetimi(metin, TOOL_CALLS) == []
    assert odul_denetimi(metin, TOOL_CALLS)


def test_dogru_odul_orani_gecer():
    assert odul_denetimi("Kampanya %8 nakit iade sağlıyor.", TOOL_CALLS) == []


def test_katalog_verisi_yoksa_odul_denetimi_sessiz():
    """`get_campaign_details` çağrılmadıysa kıyaslanacak gerçek yok."""
    sadece_analiz = [TOOL_CALLS[0]]
    assert odul_denetimi("Kampanya %25 iade sağlıyor.", sadece_analiz) == []


# --------------------------------------------------------------------------
# Tavsiye yasağı ve kampanya kaçağı
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metin", [
    "Biraz tasarruf etmenizi öneririz.",
    "Bu aboneliği iptal edebilirsiniz.",
    "Birikiminizi yatırım yapmanız daha iyi olur.",
])
def test_tavsiye_ifadeleri_yakalanir(metin):
    assert tavsiye_denetimi(metin)


def test_gozlem_cumlesi_tavsiye_sayilmaz():
    metin = "Telekom kategorisinde 6 aydır aylık ortalama 450 TL ödemeniz görünüyor."
    assert tavsiye_denetimi(metin) == []


def test_izinsiz_kampanya_yakalanir():
    sonuc = denetle("Size KMP014 kampanyasını öneriyorum.", TOOL_CALLS, KATALOG)
    assert any("izinsiz_kampanya: KMP014" in i for i in sonuc.ihlaller)


def test_onerilen_kampanya_ihlal_degil():
    sonuc = denetle("KMP001 kampanyası size uygun.", TOOL_CALLS, KATALOG)
    assert not any("izinsiz_kampanya" in i for i in sonuc.ihlaller)


# --------------------------------------------------------------------------
# mask_payload — LLM'e gidecek veri
# --------------------------------------------------------------------------


def test_pii_anahtarlari_maskelenir():
    """Şemada zaten yok ama gerçek veriye geçişte `SELECT *` kazasına karşı ağ."""
    from src.agent.guardrails import mask_payload

    kirli = {"customer_id": "C1", "tckn": "12345678901", "iban": "TR33",
             "telefon": "0555", "tutar": 100}
    temiz = mask_payload(kirli)
    assert temiz == {"customer_id": "C1", "tutar": 100}


def test_kampanya_adi_maskelenmez():
    """`name` PII listesinde ama kampanyanın adı kişinin adı değil.

    Elenirse agent kampanyayı adıyla anamaz.
    """
    from src.agent.guardrails import mask_payload

    veri = {"campaign_id": "KMP001", "name": "Market Alışverişine Nakit İade"}
    assert mask_payload(veri)["name"] == "Market Alışverişine Nakit İade"


def test_payload_kirpilir():
    from src.agent.guardrails import mask_payload

    veri = {
        "kategori_dagilimi": [{"kategori": f"k{i}"} for i in range(12)],
        "aylik_seri": [{"ay": f"2026-{i:02d}"} for i in range(1, 13)],
        "duzenli_giderler": [{"m": i} for i in range(5)],
        "olagandisi_artislar": [{"a": i} for i in range(5)],
    }
    limit = {"max_categories": 3, "max_series_months": 2,
             "max_recurring": 1, "max_anomalies": 1}
    out = mask_payload(veri, limit)
    assert len(out["kategori_dagilimi"]) == 3
    assert len(out["duzenli_giderler"]) == 1
    # Aylık seri SONDAN kırpılır: son aylar anlamlı olan.
    assert [a["ay"] for a in out["aylik_seri"]] == ["2026-11", "2026-12"]


def test_sifir_limit_alani_tamamen_bosaltir():
    """`liste[-0:]` Python'da TÜM listeyi döndürür.

    Bu yüzden "hiç gönderme" ayarı tam tersine çalışıp payload'ı büyütüyordu.
    """
    from src.agent.guardrails import mask_payload

    veri = {"aylik_seri": [{"ay": f"2026-{i:02d}"} for i in range(1, 13)]}
    assert mask_payload(veri, {"max_series_months": 0})["aylik_seri"] == []


def test_maskeleme_ic_ice_yapilarda_calisir():
    from src.agent.guardrails import mask_payload

    veri = {"oneriler": [{"campaign_id": "KMP001", "iban": "TR33", "score": 0.5}]}
    assert mask_payload(veri)["oneriler"][0] == {"campaign_id": "KMP001", "score": 0.5}
