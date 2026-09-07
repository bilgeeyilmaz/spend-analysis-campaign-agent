"""Araç katmanı testleri (LLM yok).

İki bölüm:

  1. **Dedektörler** — elle kurulmuş işlem tablolarıyla. Sentetik veriyle test
     etmek burada işe yaramaz: "düzenli gider bulundu" iddiasının doğru olup
     olmadığını ancak girdiyi kendin kurduğunda bilirsin. Her testin bir de
     NEGATİF eşi var — dedektörün ne bulmadığı, ne bulduğu kadar önemli.
  2. **Araçlar** — `opt_in_marketing` kapısı, uygunluk kuralları, PII eleme,
     skorlamanın uygun olmayan kampanyayı hiç döndürmemesi.
"""

from __future__ import annotations

import copy
import shutil
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.agent import spending
from src.agent.tools import PROFILE_EXCLUDED, PROFILE_FIELDS, TOOL_NAMES, TOOL_SCHEMAS, ToolContext
from src.data import generator
from src.features.build import build_feature_table
from src.models.propensity import train

AS_OF = pd.Timestamp("2026-08-13")
N_CUSTOMERS = 220


# --------------------------------------------------------------------------
# Elle kurulmuş işlem tabloları
# --------------------------------------------------------------------------


def _tx(kayitlar: list[dict]) -> pd.DataFrame:
    """Test işlemleri: (gun_once, tutar, kategori, merchant) -> DataFrame."""
    return pd.DataFrame([
        {
            "customer_id": "C_TEST",
            "transaction_date": AS_OF - timedelta(days=k["gun_once"]),
            "amount": float(k["tutar"]),
            "mcc_category": k.get("kategori", "market"),
            "merchant_id": k.get("merchant", "M_TEST_001"),
            "mcc_code": "5411",
            "channel": "pos",
            "installment_count": 1,
            "transaction_id": f"T{i:04d}",
        }
        for i, k in enumerate(kayitlar)
    ])


def _aylik(merchant: str, tutar: float, ay_sayisi: int, oynama: float = 0.0,
           kategori: str = "telekom", baslangic_gun: int = 5) -> list[dict]:
    """Aylık tekrarlayan ödeme üretir (30 gün arayla)."""
    return [
        {
            "gun_once": baslangic_gun + 30 * i,
            "tutar": tutar * (1 + oynama * (1 if i % 2 else -1)),
            "kategori": kategori,
            "merchant": merchant,
        }
        for i in range(ay_sayisi)
    ]


# --------------------------------------------------------------------------
# Düzenli gider dedektörü
# --------------------------------------------------------------------------


def test_duzenli_gider_bulunur():
    """Aynı işyeri, 6 ay, aylık ritim, tutar sabit -> düzenli gider."""
    tx = _tx(_aylik("M_TELEKOM_A", 450.0, ay_sayisi=6))
    sonuc = spending.duzenli_giderler(tx, AS_OF, {})
    assert len(sonuc) == 1
    assert sonuc[0]["merchant_id"] == "M_TELEKOM_A"
    assert sonuc[0]["ay_sayisi"] == 6
    assert sonuc[0]["ortalama_tutar"] == pytest.approx(450.0)


def test_duzenli_gider_tutar_oynamasi_fazlaysa_sayilmaz():
    """%15 tolerans: tutarı ikiye katlayan bir ödeme abonelik değildir."""
    tx = _tx(_aylik("M_TELEKOM_A", 450.0, ay_sayisi=6, oynama=0.40))
    assert spending.duzenli_giderler(tx, AS_OF, {}) == []


def test_duzenli_gider_az_tekrarda_sayilmaz():
    """İki ödeme "düzenli" değildir; eşik en az 3 farklı ay."""
    tx = _tx(_aylik("M_TELEKOM_A", 450.0, ay_sayisi=2))
    assert spending.duzenli_giderler(tx, AS_OF, {}) == []


def test_duzenli_gider_ritim_aylik_degilse_sayilmaz():
    """Haftalık market alışverişi düzenli gider değil — gün farkı 25-35 dışında."""
    tx = _tx([
        {"gun_once": 7 * i, "tutar": 300.0, "kategori": "market", "merchant": "M_MARKET_A"}
        for i in range(12)
    ])
    assert spending.duzenli_giderler(tx, AS_OF, {}) == []


def test_duzenli_gider_farkli_isyerleri_birlestirilmez():
    """Aynı kategoride iki ayrı işyeri iki ayrı ödemedir."""
    tx = _tx(
        _aylik("M_TELEKOM_A", 450.0, 6)
        + _aylik("M_EGLENCE_B", 120.0, 6, kategori="eglence", baslangic_gun=18)
    )
    sonuc = spending.duzenli_giderler(tx, AS_OF, {})
    assert {s["merchant_id"] for s in sonuc} == {"M_TELEKOM_A", "M_EGLENCE_B"}
    assert sonuc[0]["ortalama_tutar"] > sonuc[1]["ortalama_tutar"], "tutara göre sıralı olmalı"


def test_duzenli_gider_tek_kacan_odemeye_dayaniklidir():
    """Bir ay atlanmışsa (60 günlük boşluk) abonelik hâlâ bulunmalı.

    Gün farklarının ORTANCASI kullanılmasının sebebi bu; ortalama kullanılsaydı
    tek bir boşluk aboneliği gizlerdi.
    """
    kayitlar = _aylik("M_TELEKOM_A", 450.0, ay_sayisi=6)
    del kayitlar[2]                       # ortadaki ödeme kaçmış
    sonuc = spending.duzenli_giderler(_tx(kayitlar), AS_OF, {})
    assert len(sonuc) == 1
    assert sonuc[0]["ay_sayisi"] == 5


# --------------------------------------------------------------------------
# Olağandışı artış dedektörü
# --------------------------------------------------------------------------


def _taban_ve_sicrama(taban: float, son: float, kategori: str = "elektronik") -> pd.DataFrame:
    """3 taban penceresi (her biri `taban` TL) + son pencerede `son` TL."""
    kayitlar = []
    for pencere in (1, 2, 3):
        kayitlar.append({"gun_once": pencere * 30 + 5, "tutar": taban, "kategori": kategori})
    kayitlar.append({"gun_once": 10, "tutar": son, "kategori": kategori})
    return _tx(kayitlar)


def test_olagandisi_artis_bulunur():
    tx = _taban_ve_sicrama(taban=2000.0, son=8000.0)
    sonuc = spending.olagandisi_artislar(tx, AS_OF, {})
    assert len(sonuc) == 1
    assert sonuc[0]["kategori"] == "elektronik"
    assert sonuc[0]["donem"] == "son 30 gün"
    assert sonuc[0]["kat"] == pytest.approx(4.0)
    assert sonuc[0]["fark"] == pytest.approx(6000.0)


def test_olagandisi_artis_kucuk_tabanda_tetiklenmez():
    """Taban 1.500 TL altındaysa 4 kat artış bile anlamlı değil.

    Bu şart olmadan seyrek kategorilerdeki doğal dalgalanma müşterilerin
    %96'sını işaretliyordu.
    """
    tx = _taban_ve_sicrama(taban=200.0, son=900.0)
    assert spending.olagandisi_artislar(tx, AS_OF, {}) == []


def test_olagandisi_artis_kucuk_mutlak_farkta_tetiklenmez():
    """Oran yeterli ama fark 2.500 TL altında -> müşteriye söylemeye değmez."""
    tx = _taban_ve_sicrama(taban=1600.0, son=4000.0)
    assert spending.olagandisi_artislar(tx, AS_OF, {}) == []


def test_olagandisi_artis_taban_eksikse_tetiklenmez():
    """Taban pencerelerinin birinde harcama yoksa "artış" iddiası kurulamaz."""
    tx = _tx([
        {"gun_once": 35, "tutar": 3000.0, "kategori": "elektronik"},
        {"gun_once": 95, "tutar": 3000.0, "kategori": "elektronik"},
        # 61-90 gün penceresi boş
        {"gun_once": 10, "tutar": 12000.0, "kategori": "elektronik"},
    ])
    assert spending.olagandisi_artislar(tx, AS_OF, {}) == []


def test_olagandisi_artis_eski_pencerede_de_bulunur():
    """Sıçrama 2 ay önce olduysa da bulunmalı — üreteç son 3 aya ekiyor."""
    kayitlar = [
        {"gun_once": p * 30 + 5, "tutar": 2000.0, "kategori": "seyahat"}
        for p in (2, 3, 4)                      # 61-90, 91-120, 121-150 gün
    ]
    kayitlar.append({"gun_once": 40, "tutar": 9000.0, "kategori": "seyahat"})  # 31-60
    sonuc = spending.olagandisi_artislar(_tx(kayitlar), AS_OF, {})
    assert len(sonuc) == 1
    assert sonuc[0]["donem"] == "31-60 gün önce"


def test_olagandisi_artis_tek_islemden_mi_isaretlenir():
    """Artışın tek bir alımdan mı yayılmış harcamadan mı geldiği ayırt edilir."""
    tek = spending.olagandisi_artislar(_taban_ve_sicrama(2000.0, 8000.0), AS_OF, {})
    assert tek[0]["tip"] == "kategori_artisi"
    assert tek[0]["tek_islemden_mi"] is True

    kayitlar = [
        {"gun_once": p * 30 + 5, "tutar": 2000.0, "kategori": "elektronik"}
        for p in (1, 2, 3)
    ]
    kayitlar += [{"gun_once": 5 + i, "tutar": 1600.0, "kategori": "elektronik"} for i in range(5)]
    yaygin = spending.olagandisi_artislar(_tx(kayitlar), AS_OF, {})
    assert yaygin[0]["tek_islemden_mi"] is False


# --------------------------------------------------------------------------
# Olağandışı büyük tek alım dedektörü
# --------------------------------------------------------------------------


def _gecmis(tutar: float, adet: int, kategori: str, ilk_gun: int = 100) -> list[dict]:
    """Tarama penceresinin (90 gün) DIŞINDA tipik sepet geçmişi."""
    return [
        {"gun_once": ilk_gun + 10 * i, "tutar": tutar, "kategori": kategori}
        for i in range(adet)
    ]


def test_buyuk_alim_bulunur():
    """Tipik sepeti 400 TL olan müşterinin 9.000 TL'lik işlemi olağandışıdır."""
    tx = _tx(
        _gecmis(400.0, 10, "elektronik")
        + [{"gun_once": 12, "tutar": 9000.0, "kategori": "elektronik"}]
    )
    sonuc = spending.buyuk_alimlar(tx, AS_OF, {})
    assert len(sonuc) == 1
    assert sonuc[0]["tip"] == "buyuk_tek_alim"
    assert sonuc[0]["kat"] == pytest.approx(22.5)
    assert sonuc[0]["tipik_sepet"] == pytest.approx(400.0)


def test_buyuk_alim_esigi_gorecelidir():
    """Aynı 9.000 TL, sepeti zaten büyük olan müşteride olağandışı DEĞİLDİR.

    Mutlak eşik kullanılsaydı private segment müşteri her ay işaretlenirdi.
    """
    tx = _tx(
        _gecmis(6000.0, 10, "elektronik")
        + [{"gun_once": 12, "tutar": 9000.0, "kategori": "elektronik"}]
    )
    assert spending.buyuk_alimlar(tx, AS_OF, {}) == []


def test_buyuk_alim_kucuk_tutarda_tetiklenmez():
    """Oran yeterli ama tutar 5.000 TL altında -> söylemeye değmez."""
    tx = _tx(_gecmis(50.0, 10, "market") + [{"gun_once": 12, "tutar": 900.0, "kategori": "market"}])
    assert spending.buyuk_alimlar(tx, AS_OF, {}) == []


def test_buyuk_alim_tipik_sepeti_gecmisten_hesaplar():
    """Aday işlem kendi referansına katılmaz.

    Katılsaydı o kategoride tek işlemi olan müşteride oran her zaman 1.0 çıkar
    ve tespit hiç çalışmazdı — tam da yakalanması gereken durumda.
    """
    tx = _tx(
        _gecmis(300.0, 8, "market")               # geçmiş: başka kategori
        + [{"gun_once": 12, "tutar": 14000.0, "kategori": "seyahat"}]
    )
    sonuc = spending.buyuk_alimlar(tx, AS_OF, {})
    assert len(sonuc) == 1, "hiç geçmişi olmayan kategoride genel medyana düşmeli"
    assert sonuc[0]["kategori"] == "seyahat"


def test_buyuk_alim_kategori_basina_en_buyugu_doner():
    tx = _tx(
        _gecmis(400.0, 10, "elektronik")
        + [{"gun_once": 12, "tutar": 9000.0, "kategori": "elektronik"},
           {"gun_once": 20, "tutar": 15000.0, "kategori": "elektronik"}]
    )
    sonuc = spending.buyuk_alimlar(tx, AS_OF, {})
    assert len(sonuc) == 1
    assert sonuc[0]["tutar"] == pytest.approx(15000.0)


def test_analizde_iki_dedektor_de_yer_alir():
    """`olagandisi_artislar` iki tetikleme yolunu birleştirir, `tip` ayırır."""
    tx = _tx(
        _gecmis(400.0, 10, "elektronik")
        + [{"gun_once": 12, "tutar": 9000.0, "kategori": "elektronik"}]
    )
    sonuc = spending.analiz_et(tx, as_of=AS_OF)
    tipler = {a["tip"] for a in sonuc["olagandisi_artislar"]}
    assert "buyuk_tek_alim" in tipler


# --------------------------------------------------------------------------
# Analiz çıktısının bütünü
# --------------------------------------------------------------------------


def test_analiz_bos_islemde_cokmez():
    sonuc = spending.analiz_et(pd.DataFrame(columns=["transaction_date", "amount"]))
    assert sonuc["islem_var"] is False
    assert sonuc["kategori_dagilimi"] == []
    assert sonuc["duzenli_giderler"] == []


def test_kategori_paylari_toplami_bir():
    tx = _tx([
        {"gun_once": 5, "tutar": 600.0, "kategori": "market"},
        {"gun_once": 6, "tutar": 400.0, "kategori": "restoran"},
    ])
    kirilim = spending.kategori_kirilimi(tx)
    assert sum(k["pay"] for k in kirilim) == pytest.approx(1.0, abs=1e-3)
    assert kirilim[0]["kategori"] == "market"


def test_aylik_seri_bos_ayi_sifirla_doldurur():
    tx = _tx([{"gun_once": 5, "tutar": 1000.0}])
    seri = spending.aylik_seri(tx, AS_OF, months=12)
    assert len(seri) == 12
    assert seri[-1]["tutar"] == pytest.approx(1000.0)
    assert all(a["tutar"] == 0.0 for a in seri[:-1])


# --------------------------------------------------------------------------
# Araç katmanı (gerçek veri akışı üzerinde)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ctx(tmp_path_factory) -> ToolContext:
    """Küçük bir veri seti + özellik tablosu + eğitilmiş model ile bağlam."""
    raw_dir = tmp_path_factory.mktemp("raw")
    processed_dir = tmp_path_factory.mktemp("processed")
    model_dir = tmp_path_factory.mktemp("models")
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "campaigns.json", raw_dir / "campaigns.json")
    assert generator.main([
        "--n-customers", str(N_CUSTOMERS), "--out", str(raw_dir),
        "--as-of", "2026-08-13", "--seed", "7",
    ]) == 0

    config = copy.deepcopy(load_config())
    config["data"]["json"]["raw_dir"] = str(raw_dir)
    config["data"]["processed_dir"] = str(processed_dir)
    config["models"]["dir"] = str(model_dir)
    config["models"]["propensity"]["param_grid"] = {
        "learning_rate": [0.05], "num_leaves": [4], "n_estimators": [100],
    }
    build_feature_table(config).to_parquet(processed_dir / "customer_features.parquet", index=False)

    baglam = ToolContext(config)
    paket, _ = train(config, algorithm="hist")
    baglam._propensity = paket
    return baglam


def _musteri(ctx: ToolContext, **kosul) -> str:
    """Verilen koşulu sağlayan ilk müşterinin id'si."""
    df = ctx.source.get_customers()
    for alan, deger in kosul.items():
        df = df[df[alan] == deger]
    assert not df.empty, f"koşulu sağlayan müşteri yok: {kosul}"
    return df.iloc[0]["customer_id"]


def test_profil_pii_icermez(ctx):
    """`gender` ve `city` profil çıktısına girmez."""
    profil = ctx.get_customer_profile(_musteri(ctx))
    for alan in PROFILE_EXCLUDED:
        assert alan not in profil
    assert set(profil) <= set(PROFILE_FIELDS)


def test_bilinmeyen_musteri_hata_verir(ctx):
    with pytest.raises(KeyError):
        ctx.get_customer_profile("C_YOK")


def test_opt_in_yoksa_uygunluk_ilk_kurala_takilir(ctx):
    """Yasal kapı: izin yoksa diğer kurallara BAKILMAZ."""
    cid = _musteri(ctx, opt_in_marketing=False)
    sonuc = ctx.check_eligibility(cid, "KMP001")
    assert sonuc["uygun"] is False
    assert sonuc["basarisiz_kurallar"] == ["opt_in_marketing"]


def test_opt_in_yoksa_hic_oneri_donmez(ctx):
    """Skor ne olursa olsun izinsiz müşteriye kampanya önerilmez."""
    cid = _musteri(ctx, opt_in_marketing=False)
    sonuc = ctx.score_campaigns(cid)
    assert sonuc["oneriler"] == []
    assert sonuc["uygun_kampanya_sayisi"] == 0
    assert "opt_in_marketing" in sonuc["gerekce"]


def test_suresi_gecmis_kampanya_uygun_degil(ctx):
    """KMP013 negatif fixture: süresi 30 Haziran'da doldu."""
    cid = _musteri(ctx, opt_in_marketing=True)
    sonuc = ctx.check_eligibility(cid, "KMP013")
    assert sonuc["uygun"] is False
    assert "kampanya_tarihi" in sonuc["basarisiz_kurallar"]


def test_butcesi_biten_kampanya_uygun_degil(ctx):
    """KMP014 negatif fixture: remaining_budget = 0."""
    cid = _musteri(ctx, opt_in_marketing=True)
    sonuc = ctx.check_eligibility(cid, "KMP014")
    assert sonuc["uygun"] is False
    assert "butce_tukendi" in sonuc["basarisiz_kurallar"]


def test_skorlama_uygun_olmayan_kampanyayi_dondurmez(ctx):
    """Araç kendi içinde eliyor; LLM'in uygunluğu atlaması mümkün olmasın."""
    cid = _musteri(ctx, opt_in_marketing=True)
    sonuc = ctx.score_campaigns(cid, top_k=20)
    onerilen = {o["campaign_id"] for o in sonuc["oneriler"]}
    for kampanya_id in onerilen:
        assert ctx.check_eligibility(cid, kampanya_id)["uygun"] is True
    assert "KMP013" not in onerilen and "KMP014" not in onerilen


def test_skorlar_sirali_ve_olasilik(ctx):
    cid = _musteri(ctx, opt_in_marketing=True)
    oneriler = ctx.score_campaigns(cid, top_k=5)["oneriler"]
    skorlar = [o["score"] for o in oneriler]
    assert skorlar == sorted(skorlar, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in skorlar)
    assert [o["sira"] for o in oneriler] == list(range(1, len(oneriler) + 1))


def test_analiz_katalogdan_bagimsiz_calisir(ctx):
    """Harcama analizi kampanya kataloğuna bakmaz; tek başına anlamlı çıktı verir."""
    cid = _musteri(ctx, opt_in_marketing=False)     # hiç kampanya alamayacak müşteri
    analiz = ctx.analyze_spending(cid)
    assert analiz["islem_var"] is True
    assert analiz["kategori_dagilimi"], "izinsiz müşteride de analiz üretilmeli"
    assert "kampanya" not in str(analiz).lower()


def test_kampanya_detayi_json_uyumlu(ctx):
    detay = ctx.get_campaign_details("KMP001")
    assert detay["campaign_id"] == "KMP001"
    assert isinstance(detay["valid_from"], str)
    assert isinstance(detay["aktif"], bool)


# --------------------------------------------------------------------------
# LLM'e verilecek imzalar
# --------------------------------------------------------------------------


def test_tool_schemas_araclarla_ortusuyor(ctx):
    for ad in TOOL_NAMES:
        assert hasattr(ctx, ad), f"{ad} şemada var ama ToolContext'te yok"
    assert len(TOOL_SCHEMAS) == 5


def test_call_bilinmeyen_araci_reddeder(ctx):
    with pytest.raises(ValueError, match="Bilinmeyen araç"):
        ctx.call("drop_database", {})


def test_call_dagiticisi_calisir(ctx):
    cid = _musteri(ctx)
    assert ctx.call("get_customer_profile", {"customer_id": cid})["customer_id"] == cid


# --------------------------------------------------------------------------
# Dönemsel karşılaştırma ve üye işyeri kırılımı
# --------------------------------------------------------------------------


def test_karsilastirma_esit_uzunlukta_onceki_donemi_alir():
    """3 aylık pencere 1 önceki aya değil, önceki 3 aya kıyaslanır.

    Kısa bir referans dönem farkı olduğundan büyük gösterir; kullanıcı "market
    harcamam üçe katlandı" diye okur, oysa yalnız pencere uzunlukları farklıdır.
    """
    tx = _tx(
        [{"gun_once": 10, "tutar": 300, "kategori": "market"}]          # bu dönem
        + [{"gun_once": 100, "tutar": 100, "kategori": "market"},        # önceki dönem
           {"gun_once": 130, "tutar": 200, "kategori": "market"}]
    )

    satir = next(k for k in spending.kategori_karsilastirma(tx, AS_OF, months=3)
                 if k["kategori"] == "market")

    assert satir["tutar"] == 300.0
    assert satir["onceki_tutar"] == 300.0      # 100 + 200, aynı uzunlukta pencere
    assert satir["degisim_orani"] == 0.0


def test_karsilastirma_bu_donem_bitmis_kategoriyi_de_gosteriyor():
    """Sadece bu dönemin kategorilerine bakmak "market harcamam bitti" bilgisini
    sessizce yutardı; kategoriler iki dönemin birleşimi."""
    tx = _tx([
        {"gun_once": 10, "tutar": 500, "kategori": "restoran"},
        {"gun_once": 100, "tutar": 400, "kategori": "market"},
    ])

    kategoriler = {k["kategori"]: k for k in spending.kategori_karsilastirma(tx, AS_OF, 3)}

    assert kategoriler["market"]["tutar"] == 0.0
    assert kategoriler["market"]["degisim_orani"] == -1.0


def test_karsilastirma_yeni_kategoride_oran_tanimsiz():
    """Önceki dönem sıfırsa "sonsuz artış" yazmak yerine None: arayüz "yeni" der."""
    tx = _tx([{"gun_once": 10, "tutar": 500, "kategori": "seyahat"}])

    satir = spending.kategori_karsilastirma(tx, AS_OF, 3)[0]

    assert satir["onceki_tutar"] == 0.0
    assert satir["degisim_orani"] is None


def test_uye_isyerleri_tutara_gore_sirali_ve_sinirli():
    tx = _tx([
        {"gun_once": 5, "tutar": 100, "merchant": "M_A"},
        {"gun_once": 6, "tutar": 900, "merchant": "M_B"},
        {"gun_once": 7, "tutar": 400, "merchant": "M_C"},
    ])

    ilk = spending.ilk_uye_isyerleri(tx, AS_OF, months=3, n=2)

    assert [m["merchant_id"] for m in ilk] == ["M_B", "M_C"]
    assert ilk[0]["tutar"] == 900.0


def test_uye_isyeri_kategorisi_en_sik_kategoridir():
    """Bir işyeri birden çok kategoride görünebilir; etiket baskın olanı olmalı."""
    tx = _tx([
        {"gun_once": 5, "tutar": 100, "merchant": "M_A", "kategori": "market"},
        {"gun_once": 6, "tutar": 100, "merchant": "M_A", "kategori": "market"},
        {"gun_once": 7, "tutar": 100, "merchant": "M_A", "kategori": "diger"},
    ])

    assert spending.ilk_uye_isyerleri(tx, AS_OF, 3)[0]["kategori"] == "market"


def test_islemsiz_musteride_yeni_alanlar_bos():
    sonuc = spending.analiz_et(pd.DataFrame(columns=["transaction_date", "amount"]))

    assert sonuc["kategori_karsilastirma"] == []
    assert sonuc["ilk_uye_isyerleri"] == []


def test_pencere_as_of_sonrasini_saymiyor():
    """`as_of` geçmiş bir tarihe sabitlendiğinde sonraki işlemler döneme girmemeli.

    Üst sınır olmadığında `config.features.as_of_date` ile geçmişe dönük analiz
    yapmak tutarları sessizce şişiriyordu — ve o ayar tam olarak geçmişe dönük
    yeniden üretim için var.
    """
    tx = _tx([
        {"gun_once": 10, "tutar": 100},      # dönem içi
        {"gun_once": -5, "tutar": 900},      # as_of'tan SONRA
    ])

    sonuc = spending.analiz_et(tx, as_of=AS_OF, months=3)

    assert sonuc["toplam_harcama"] == 100.0
    assert sonuc["islem_adedi"] == 1


@pytest.mark.parametrize("donem, beklenen", [
    ("2026-06", "Haziran 2026"),
    ("2026-01", "Ocak 2026"),
    ("2025-12", "Aralık 2025"),
    ("bozuk", "bozuk"),
])
def test_ay_adi_turkce(donem, beklenen):
    assert spending.ay_adi(donem) == beklenen


def test_aylik_seri_turkce_ay_adi_tasiyor():
    """LLM'e yalnız "2026-06" verildiğinde cevabı "June 2026" diye İngilizce
    yazıyordu; doğru biçimi promptla değil veriyle vermek daha güvenilir."""
    tx = _tx([{"gun_once": 10, "tutar": 100}])

    seri = spending.aylik_seri(tx, AS_OF, months=3)

    assert all("ay_adi" in ay for ay in seri)
    assert any(ay["ay_adi"].split()[0] in spending.AY_ADLARI for ay in seri)
