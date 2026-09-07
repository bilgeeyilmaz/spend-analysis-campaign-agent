"""Arayüzün saf fonksiyonları — palet, biçimlendirme, satır ve metin üretimi.

Streamlit sayfasının kendisi burada çalıştırılmıyor (o, canlı servis ister);
sınanan şey arayüzün karar verdiği kısım. Grafik/renk testleri "güzel mi" diye
sormaz, tasarım kararlarını kilitler — bunlar sessizce bozulduğunda kimse fark
etmez, ekran yine çizilir, sadece kötü çizilir ya da okunmaz hâle gelir.

Palet testleri gerçekten **ölçüm** yapar: kontrast oranları hesaplanır ve
eşiklere karşı sınanır. Marka sarısının beyaz zeminde veri işareti olarak
kullanılamayacağı bu projede ölçülmüş bir bulgudur, yorum değil.
"""

from __future__ import annotations

import pytest

from src.features.categories import CATEGORIES
from ui.streamlit_app import (
    DURUM_ROZETI,
    ARTIK_RENGI,
    KATEGORI_RENGI,
    ISARET_ESIGI,
    KATEGORI_GORUNUM,
    METIN_ESIGI,
    PALET,
    aylik_grafik,
    butce_durumu,
    degisim_rozeti,
    denetim_durumu,
    gorunum,
    halka_dilimleri,
    hero_karti,
    isyeri_adi,
    kategori_tonlari,
    halka_grafigi,
    hazir_soru_secildi,
    kampanya_kartlari,
    kategori_listesi,
    kategori_satirlari,
    kontrast_orani,
    one_cikanlar,
    odul_metni,
    onerilen_kampanyalar,
    onerilen_sorular,
    rgba,
    tema,
    tl,
    trend_rozeti,
    uye_isyeri_listesi,
)

DAGILIM = [
    {"kategori": "market", "tutar": 40771.53, "pay": 0.4884, "islem_adedi": 37},
    {"kategori": "akaryakit", "tutar": 10000.0, "pay": 0.12, "islem_adedi": 8},
    {"kategori": "saglik", "tutar": 9000.0, "pay": 0.11, "islem_adedi": 4},
]
SERI = [
    {"ay": "2026-06", "tutar": 28376.0, "islem_adedi": 25},
    {"ay": "2026-07", "tutar": 24498.0, "islem_adedi": 22},
    {"ay": "2026-08", "tutar": 12000.0, "islem_adedi": 11},
]
ANALIZ = {
    "islem_var": True,
    "ilk_uc_kategori": ["market", "akaryakit", "saglik"],
    "duzenli_giderler": [
        {"kategori": "eglence", "ortalama_tutar": 214.08, "ay_sayisi": 6, "tipik_gun": 6},
    ],
    "olagandisi_artislar": [],
    "trend": {"degisim_orani": -0.23, "son_30_gun": 8022.0},
}


@pytest.fixture
def renk() -> dict:
    return PALET["light"]


# -- biçimlendirme ---------------------------------------------------------


@pytest.mark.parametrize("deger, beklenen", [
    (0, "0 TL"),
    (1234.4, "1.234 TL"),
    (28729.76, "28.730 TL"),
    (1234567.8, "1.234.568 TL"),
])
def test_tl_turkce_ayraclarla_biciimlendiriyor(deger, beklenen):
    """Sunum Türkçe: binlik ayracı nokta. Varsayılan f-string virgül koyar."""
    assert tl(deger) == beklenen


def test_tema_calisma_zamani_disinda_acik_temaya_duser():
    """`st.context` yalnız Streamlit çalışırken okunabilir; patlamamalı."""
    assert tema() == PALET["light"]


def test_dolgu_rengi_seri_renginden_turetiliyor():
    assert rgba("#A67C00", 0.10) == "rgba(166,124,0,0.1)"


# -- kurumsal palet: renkler ölçülerek seçildi, gözle değil -----------------


@pytest.mark.parametrize("mod", ["light", "dark"])
def test_veri_isareti_renkleri_kontrasti_geciyor(mod):
    """Çubuk ve çizgi renkleri kendi zeminlerinde en az 3:1 (WCAG grafik nesnesi)."""
    p = PALET[mod]
    for rol in ("seri", "vurgu"):
        oran = kontrast_orani(p[rol], p["yuzey"])
        assert oran >= ISARET_ESIGI, f"{mod}/{rol} kontrastı {oran:.2f}:1"


@pytest.mark.parametrize("mod", ["light", "dark"])
def test_metin_renkleri_kontrasti_geciyor(mod):
    p = PALET[mod]
    for rol in ("ink", "ikincil"):
        oran = kontrast_orani(p[rol], p["yuzey"])
        assert oran >= METIN_ESIGI, f"{mod}/{rol} kontrastı {oran:.2f}:1"


def test_marka_sarisi_acik_zeminde_veri_isareti_degil():
    """Projenin ölçülmüş renk tuzağı.

    Marka sarısı beyaz zeminde ~1.46:1 verir — çubuk ya da çizgi olarak
    kullanılırsa okunmaz. Biri "daha kurumsal görünsün" diye seri rengini sarıya
    çevirirse burada patlar.
    """
    acik = PALET["light"]

    assert kontrast_orani(acik["marka"], acik["yuzey"]) < ISARET_ESIGI
    assert acik["seri"] != acik["marka"]


def test_marka_seridi_uzerindeki_yazi_okunuyor():
    """Sarı dolgunun üstündeki yazı siyah olduğu sürece sarı serbest."""
    for mod in ("light", "dark"):
        p = PALET[mod]
        assert kontrast_orani(p["marka_ink"], p["marka"]) >= METIN_ESIGI


def test_koyu_temada_marka_sarisi_kullanilabiliyor():
    """Aynı sarı koyu zeminde 12:1'in üstünde — orada seri rengi olarak serbest."""
    koyu = PALET["dark"]

    assert koyu["seri"] == koyu["marka"]
    assert kontrast_orani(koyu["seri"], koyu["yuzey"]) > 10


@pytest.mark.parametrize("mod", ["light", "dark"])
def test_kart_zemini_de_metni_tasiyabiliyor(mod):
    """Satırlar `kart` zemini üstünde duruyor; kontrast orada da tutmalı."""
    p = PALET[mod]
    assert kontrast_orani(p["ink"], p["kart"]) >= METIN_ESIGI
    assert kontrast_orani(p["seri"], p["kart"]) >= ISARET_ESIGI


# -- kategori görünümü -----------------------------------------------------


def test_her_kategorinin_simgesi_ve_turkce_adi_var():
    """Şema kimlikleri İngilizce ('online_alisveris'); kullanıcı onları görmemeli.
    Yeni kategori eklenip görünümü unutulursa ekrana ham kimlik düşer."""
    eksik = sorted(set(CATEGORIES) - set(KATEGORI_GORUNUM))
    assert not eksik, f"görünümü tanımlanmamış kategori: {eksik}"


def test_bilinmeyen_kategori_ekrani_bozmuyor():
    simge, ad = gorunum("kripto_varlik")

    assert simge and ad == "Kripto varlik"


# -- kategori satırları ----------------------------------------------------


def test_satirlar_buyukten_kucuge_sirali():
    satirlar = kategori_satirlari(DAGILIM)

    assert [s["kategori"] for s in satirlar] == ["market", "akaryakit", "saglik"]


def test_cubuk_genisligi_yazan_payla_ayni():
    """Ölçülmüş hata: çubuk en büyük kategoriye göre ölçeklenince satırda "%24"
    yazarken çubuk tam dolu görünüyordu. Çubuk, yazan sayının görsel karşılığı."""
    satirlar = kategori_satirlari(DAGILIM)

    for s in satirlar:
        assert s["genislik"] == pytest.approx(s["pay"] * 100, abs=0.1)


def test_cok_kucuk_pay_gorunur_kaliyor():
    """%0.2'lik kategori çubuğu yokmuş gibi görünmemeli."""
    kucuk = [{"kategori": "saglik", "tutar": 20.0, "pay": 0.002, "islem_adedi": 1}]

    assert kategori_satirlari(kucuk)[0]["genislik"] >= 1.5


def test_satirlar_turkce_ad_ve_simge_tasiyor():
    ilk = kategori_satirlari(DAGILIM)[0]

    assert ilk["ad"] == "Market"
    assert ilk["simge"]


def test_bos_dagilim_satir_uretmiyor():
    assert kategori_satirlari([]) == []


def test_secili_kategori_vurgu_rengiyle_ciziliyor(renk):
    html_metin = kategori_listesi(DAGILIM, renk, secili="market")

    assert f"background:{renk['vurgu']}" in html_metin
    assert f"background:{renk['seri']}" in html_metin       # diğerleri normal


def test_secim_yokken_tum_cubuklar_seri_renginde(renk):
    html_metin = kategori_listesi(DAGILIM, renk)

    assert renk["vurgu"] not in html_metin.split("satir-dolgu")[1]


# -- trend rozeti ----------------------------------------------------------


def test_artis_ve_azalis_farkli_gosteriliyor(renk):
    artis = trend_rozeti(0.23, renk)
    azalis = trend_rozeti(-0.23, renk)

    assert "▲" in artis and "%23" in artis
    assert "▼" in azalis
    assert renk["artis"] in artis and renk["azalis"] in azalis


def test_trend_yoksa_rozet_yok(renk):
    assert trend_rozeti(None, renk) == ""


# -- aylık seri ------------------------------------------------------------


def test_aylik_grafik_cizgi_ve_isaretci_olculeri(renk):
    """İnce çizgi (2px) + okunur işaretçi (>=8px)."""
    iz = aylik_grafik(SERI, renk).data[0]

    assert iz.mode == "lines+markers"
    assert iz.line.width == 2
    assert iz.marker.size >= 8


def test_aylik_grafik_noktalara_sayi_yazmiyor(renk):
    """Her noktaya değer yazmak çizgiyi okunmaz yapar; sayı hover'da."""
    iz = aylik_grafik(SERI, renk).data[0]

    assert iz.text is None
    assert "%{y" in iz.hovertemplate


def test_aylik_grafik_zemini_seffaf(renk):
    """Opak zemin, koyu temada beyaz bir kutu olarak sırıtır."""
    fig = aylik_grafik(SERI, renk)

    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_aylik_grafik_lejantsiz(renk):
    """Tek seri: lejant kutusu gürültüdür, başlık zaten seriyi söylüyor."""
    assert aylik_grafik(SERI, renk).layout.showlegend is False


# -- denetim durumu --------------------------------------------------------


def _cevap(**ustune) -> dict:
    temel = {"answer": "metin", "provider": "groq", "tool_calls": [],
             "iterations": 2, "fallback": False, "uyarilar": []}
    return {**temel, **ustune}


def test_temiz_cevap_basarili_gosterilir():
    seviye, mesaj = denetim_durumu(_cevap())

    assert seviye == "ok"
    assert "geçti" in mesaj


def test_duzeltilmis_cevap_hata_olarak_gosterilmez():
    """Ölçülmüş gerçek durum: uyarı listesi dolu ama fallback False.

    Orkestratör ihlali yakalayıp metni düzelttirmiş ve düzeltilmiş metin
    denetimden geçmiştir. Bunu kırmızı hata olarak göstermek, çalışan
    guardrail'i arıza gibi sunar — demoda tam ters mesaj verir.
    """
    seviye, mesaj = denetim_durumu(
        _cevap(uyarilar=["izlenemeyen_sayi: [6226.0]"], fallback=False)
    )

    assert seviye == "duzeltildi"
    assert "denetimden geçti" in mesaj
    assert "6226.0" in mesaj


def test_yedege_dusen_cevap_ayirt_edilir():
    seviye, mesaj = denetim_durumu(_cevap(uyarilar=["max_iterations"], fallback=True))

    assert seviye == "yedek"
    assert "şablon" in mesaj and "max_iterations" in mesaj


# -- ajan sekmesi: hazır sorular ve öne çıkanlar ---------------------------


def test_hazir_sorular_veriden_turetiliyor():
    """Düzenli gideri olana o soru önerilir, olmayana önerilmez."""
    varsa = onerilen_sorular(ANALIZ)
    yoksa = onerilen_sorular({**ANALIZ, "duzenli_giderler": []})

    assert "Düzenli ödemelerim neler?" in varsa
    assert "Düzenli ödemelerim neler?" not in yoksa


def test_hazir_sorular_neden_diye_sormuyor():
    """Ajan tarif eder, yorum yapmaz: "neden arttı" onu sebep uydurmaya davet eder."""
    for soru in onerilen_sorular(ANALIZ):
        assert "neden" not in soru.lower()


def test_hazir_sorular_dort_ile_sinirli():
    zengin = {**ANALIZ, "olagandisi_artislar": [{"kategori": "elektronik", "tutar": 9000}]}

    assert len(onerilen_sorular(zengin)) <= 4


def test_hazir_soru_turkce_kategori_adi_kullaniyor():
    """Soruda ham şema kimliği ('online_alisveris') görünmemeli."""
    sorular = onerilen_sorular({**ANALIZ, "ilk_uc_kategori": ["online_alisveris"]})

    assert any("Online alışveriş harcamam" in s for s in sorular)


def test_one_cikanlar_duzenli_gideri_cumleye_ceviriyor():
    satirlar = one_cikanlar(ANALIZ)

    assert any("Eğlence" in s and "214 TL" in s for s in satirlar)


def test_one_cikanlar_kayda_deger_trendi_soyluyor():
    """%15 altındaki değişim gürültüdür, satır açmaya değmez."""
    buyuk = one_cikanlar(ANALIZ)
    kucuk = one_cikanlar({**ANALIZ, "trend": {"degisim_orani": -0.03, "son_30_gun": 1.0}})

    assert any("azalmış" in s for s in buyuk)
    assert not any("azalmış" in s or "artmış" in s for s in kucuk)


def test_one_cikanlar_bos_analizde_patlamiyor():
    assert one_cikanlar({"islem_var": True}) == []


# -- hazır soru seçiminin tüketilmesi ---------------------------------------


def test_hazir_soru_secimi_tuketiliyor():
    """Ölçülmüş hata: `st.pills` seçimi kalıcıdır.

    Sıfırlanmazsa, soru sorulup sayfa yeniden çalıştığında pill hâlâ seçili
    döner, soru tekrar tetiklenir ve arayüz sonsuz döngüye girer (canlı denemede
    sayfa 300 sn'de tamamlanmadı). Yazarak sorma yolu bu hatadan etkilenmiyordu.
    """
    durum = {"hazir_C1": "Düzenli ödemelerim neler?"}

    hazir_soru_secildi("hazir_C1", durum)

    assert durum["bekleyen_soru"] == "Düzenli ödemelerim neler?"
    assert durum["hazir_C1"] is None, "seçim sıfırlanmadı — döngü riski"


def test_secim_yokken_soru_uretilmiyor():
    durum = {"hazir_C1": None}

    hazir_soru_secildi("hazir_C1", durum)

    assert "bekleyen_soru" not in durum


def test_ayni_soru_ikinci_kez_sorulabiliyor():
    """Seçim tüketildiği için pill'e tekrar tıklamak yeniden soru üretir."""
    durum = {"k": "Aynı soru"}

    hazir_soru_secildi("k", durum)
    assert durum["bekleyen_soru"] == "Aynı soru"

    durum["k"] = "Aynı soru"
    durum.pop("bekleyen_soru")
    hazir_soru_secildi("k", durum)
    assert durum["bekleyen_soru"] == "Aynı soru"


# -- dönem karşılaştırma rozeti ve bütçe -----------------------------------


KARSILASTIRMA = [
    {"kategori": "market", "tutar": 40771.53, "onceki_tutar": 20000.0,
     "fark": 20771.53, "degisim_orani": 1.04},
    {"kategori": "akaryakit", "tutar": 10000.0, "onceki_tutar": 10200.0,
     "fark": -200.0, "degisim_orani": -0.02},
    {"kategori": "saglik", "tutar": 9000.0, "onceki_tutar": 0.0,
     "fark": 9000.0, "degisim_orani": None},
]


def test_satirlar_degisim_bilgisini_tasiyor():
    satirlar = {s["kategori"]: s for s in kategori_satirlari(DAGILIM, KARSILASTIRMA)}

    assert satirlar["market"]["degisim"] == 1.04
    assert satirlar["saglik"]["yeni"] is True


def test_kucuk_degisim_rozetlenmiyor(renk):
    """Her satıra ok koymak listeyi gürültüye boğar, gerçek hareketi görünmez yapar."""
    assert degisim_rozeti(0.02, False, renk) == ""
    assert degisim_rozeti(-0.02, False, renk) == ""


def test_artis_ve_azalis_rozetleri_ayirt_ediliyor(renk):
    assert "▲" in degisim_rozeti(1.04, False, renk)
    assert "▼" in degisim_rozeti(-0.40, False, renk)
    assert renk["artis"] in degisim_rozeti(1.04, False, renk)


def test_onceki_donemde_olmayan_kategori_yeni_diye_isaretleniyor(renk):
    """Sıfıra bölme "sonsuz artış" değil, "yeni" demektir."""
    assert "yeni" in degisim_rozeti(None, True, renk)


def test_karsilastirma_verilmezse_rozet_cikmiyor(renk):
    """Eski çağrı biçimi bozulmamalı: karşılaştırma opsiyonel."""
    html_metin = kategori_listesi(DAGILIM, renk)

    assert "rozet" not in html_metin


def test_butce_dolulugu_hesaplaniyor():
    durum = butce_durumu(harcanan=750.0, limit=1000)

    assert durum["genislik"] == 75.0
    assert durum["asildi"] is False
    assert durum["kalan"] == 250.0


def test_asilan_butce_isaretleniyor_ve_cubuk_tasmiyor():
    durum = butce_durumu(harcanan=1500.0, limit=1000)

    assert durum["asildi"] is True
    assert durum["genislik"] == 100.0, "çubuk %100'ü aşarsa yerleşim bozulur"


def test_limit_yoksa_butce_paneli_bos():
    """Sistem limit ÖNERMEZ — limit yoksa gösterilecek bir doluluk da yoktur."""
    assert butce_durumu(500.0, None) is None
    assert butce_durumu(500.0, 0) is None


def test_uye_isyerleri_kategoriye_gore_suzuluyor(renk):
    isyerleri = [
        {"merchant_id": "M_A", "kategori": "market", "tutar": 900.0, "islem_adedi": 3},
        {"merchant_id": "M_B", "kategori": "seyahat", "tutar": 500.0, "islem_adedi": 1},
    ]

    hepsi = uye_isyeri_listesi(isyerleri, renk)
    market = uye_isyeri_listesi(isyerleri, renk, kategori="market")

    assert "M_A" in hepsi and "M_B" in hepsi
    assert "M_A" in market and "M_B" not in market


def test_bos_isyeri_listesi_cokmuyor(renk):
    assert "işlem yok" in uye_isyeri_listesi([], renk)


# -- halka grafiği ---------------------------------------------------------


@pytest.mark.parametrize("mod", ["light", "dark"])
def test_kategori_renklerinin_hepsi_kontrasti_geciyor(mod):
    """Her dilim kendi zemininde ayırt edilebilmeli."""
    for kategori, renk_kodu in {**KATEGORI_RENGI[mod],
                                "_artik": ARTIK_RENGI[mod]}.items():
        oran = kontrast_orani(renk_kodu, PALET[mod]["yuzey"])
        assert oran >= ISARET_ESIGI, f"{mod}/{kategori} kontrastı {oran:.2f}:1"


@pytest.mark.parametrize("mod", ["light", "dark"])
def test_her_kategorinin_sabit_bir_rengi_var(mod):
    eksik = sorted(set(CATEGORIES) - set(KATEGORI_RENGI[mod]))
    assert not eksik, f"rengi tanımlanmamış kategori: {eksik}"


@pytest.mark.parametrize("mod", ["light", "dark"])
def test_ayni_kategori_farkli_musteride_ayni_renk(mod):
    """Ölçülmüş hata: renk sıralamaya göre veriliyordu, o yüzden market bir
    müşteride kehribar, diğerinde gri çıkıyordu. Renk kimliği takip etmeli."""
    musteri_a = [{"kategori": "market", "tutar": 900.0, "pay": 0.9, "islem_adedi": 3},
                 {"kategori": "seyahat", "tutar": 100.0, "pay": 0.1, "islem_adedi": 1}]
    musteri_b = [{"kategori": "seyahat", "tutar": 900.0, "pay": 0.9, "islem_adedi": 2},
                 {"kategori": "market", "tutar": 100.0, "pay": 0.1, "islem_adedi": 1}]

    a = kategori_tonlari(musteri_a, mod)
    b = kategori_tonlari(musteri_b, mod)

    assert a["market"] == b["market"]
    assert a["seyahat"] == b["seyahat"]
    assert a["market"] != a["seyahat"]


def test_halka_en_buyuk_besi_alip_kalani_topluyor():
    """12 dilimlik halka okunmaz; kalan tek dilimde toplanır."""
    dagilim = [{"kategori": k, "tutar": 100 - i, "pay": 0.1, "islem_adedi": 1}
               for i, k in enumerate(CATEGORIES)]

    dilimler = halka_dilimleri(dagilim, n=5)

    assert len(dilimler) == 6
    assert dilimler[-1]["ad"] == f"Diğer {len(CATEGORIES) - 5} kategori"
    assert dilimler[-1]["tutar"] == sum(100 - i for i in range(5, len(CATEGORIES)))


def test_halka_az_kategoride_diger_dilimi_acmiyor():
    dilimler = halka_dilimleri(DAGILIM, n=5)

    assert [d["ad"] for d in dilimler] == ["Market", "Akaryakıt", "Sağlık"]


def test_halka_ortasinda_toplam_yaziyor(renk):
    fig = halka_grafigi(DAGILIM, 59771.53, renk, "light")

    assert fig.data[0].hole == 0.62
    assert "59.772 TL" in fig.layout.annotations[0].text


def test_halka_dilim_ustune_yazi_basmiyor(renk):
    """Dilim üstü etiket halkayı boğar; değerler hover'da ve yandaki listede."""
    assert halka_grafigi(DAGILIM, 100.0, renk, "light").data[0].textinfo == "none"


# -- kampanya kartları -----------------------------------------------------


KATALOG = [{
    "campaign_id": "KMP001", "name": "Market Alışverişine Nakit İade",
    "description": "Anlaşmalı marketlerde %8 nakit iade.", "reward_type": "cashback",
    "reward_value": 8.0, "reward_unit": "yuzde", "min_spend": 750.0,
    "valid_to": "2026-09-30", "target_categories": ["market"],
}]
CEVAP_ONERI = {
    "answer": "...", "provider": "groq", "fallback": False, "uyarilar": [], "iterations": 3,
    "tool_calls": [
        {"name": "analyze_spending", "arguments": {}, "result": {}},
        {"name": "score_campaigns", "arguments": {},
         "result": {"oneriler": [{"campaign_id": "KMP001", "name": "Market...",
                                  "score": 0.58, "sira": 1}]}},
    ],
}


def test_kart_skor_ile_katalogu_birlestiriyor():
    """Skorlama aracı ödül/asgari harcama döndürmüyor — o alanlar katalogdan gelir."""
    kart = onerilen_kampanyalar(CEVAP_ONERI, KATALOG)[0]

    assert kart["ad"] == "Market Alışverişine Nakit İade"
    assert kart["odul"] == "%8"
    assert kart["min_spend"] == 750.0


def test_skorlama_cagrilmamissa_kart_yok():
    """Serbest sohbet cevabında kampanya kartı çıkmamalı."""
    sadece_analiz = {**CEVAP_ONERI, "tool_calls": [
        {"name": "analyze_spending", "arguments": {}, "result": {}}]}

    assert onerilen_kampanyalar(sadece_analiz, KATALOG) == []


def test_katalogda_olmayan_kampanya_karti_cokmuyor():
    """Katalog ile skor listesi ayrı uçlardan geliyor; biri eksik kalabilir."""
    kart = onerilen_kampanyalar(CEVAP_ONERI, [])[0]

    assert kart["campaign_id"] == "KMP001"
    assert kart["odul"] == "—"


def test_kart_html_odulu_marka_renginde_gosteriyor(renk):
    metin = kampanya_kartlari(onerilen_kampanyalar(CEVAP_ONERI, KATALOG), renk)

    assert "%8" in metin
    assert renk["marka"] in metin


@pytest.mark.parametrize("deger, birim, beklenen", [
    (8.0, "yuzde", "%8"),
    (100.0, "tutar", "100 TL"),
    (6.0, "taksit_sayisi", "6 taksit"),
    (None, None, "—"),
])
def test_odul_kendi_birimiyle_yaziliyor(deger, birim, beklenen):
    """Ölçülmüş hata: yüzde olmayan her şeyi TL saymak "6 taksit" kampanyasını
    "6 TL" diye gösteriyordu — hem yanlış hem kampanyayı değersiz gösteren."""
    assert odul_metni(deger, birim) == beklenen


def test_taksit_kampanyasi_kartta_dogru_gorunuyor():
    katalog = [{**KATALOG[0], "reward_type": "taksit", "reward_value": 6.0,
                "reward_unit": "taksit_sayisi"}]

    assert onerilen_kampanyalar(CEVAP_ONERI, katalog)[0]["odul"] == "6 taksit"


def test_denetim_ayrintisi_musteriye_gosterilen_metinde_degil():
    """`izlenemeyen_sayi: [2263.0]` müşterinin okuyacağı bir cümle değil.

    Kısa rozet ekranda, teknik ayrıntı "Gerekçe" kutusunun içinde kalır.
    """
    for seviye in ("ok", "duzeltildi", "yedek"):
        simge, etiket = DURUM_ROZETI[seviye]
        assert simge and etiket
        assert "_" not in etiket, "rozet metninde teknik ihlal adı görünüyor"


@pytest.mark.parametrize("kimlik, kategori, beklenen", [
    ("M_ELEKTRONIK_018", "elektronik", "Elektronik #018"),
    ("M_ONLINE_ALISVERIS_004", "online_alisveris", "Online alışveriş #004"),
    ("ACME MARKET", "market", "ACME MARKET"),        # gerçek ad geldiyse dokunma
])
def test_isyeri_adi_okunur_hale_getiriliyor(kimlik, kategori, beklenen):
    """Sentetik veride işyeri adı yok; ham kimlik ekranda anlamsız görünüyordu."""
    assert isyeri_adi(kimlik, kategori) == beklenen


def test_isyeri_cubugu_kategorisinin_halka_tonunu_kullaniyor(renk):
    tonlar = kategori_tonlari(DAGILIM, "light")
    isyerleri = [{"merchant_id": "M_MARKET_001", "kategori": "market",
                  "tutar": 900.0, "islem_adedi": 3}]

    metin = uye_isyeri_listesi(isyerleri, renk, tonlar=tonlar)

    assert tonlar["market"] in metin
    assert tonlar["market"] == KATEGORI_RENGI["light"]["market"]


def test_toplanan_dilim_gercek_diger_kategorisiyle_karismiyor():
    """Veri setinde `diger` adında gerçek bir kategori var; toplanan artık dilime
    de "Diğer" demek efsanede iki ayrı "Diğer" üretiyordu."""
    dagilim = [{"kategori": k, "tutar": 100 - i, "pay": 0.1, "islem_adedi": 1}
               for i, k in enumerate(["market", "diger", "akaryakit", "restoran",
                                      "giyim", "seyahat", "telekom"])]

    adlar = [d["ad"] for d in halka_dilimleri(dagilim, n=5)]

    assert len(set(adlar)) == len(adlar), f"efsanede çakışan ad var: {adlar}"
    assert "Diğer harcamalar" in adlar          # gerçek kategori
    assert adlar[-1] == "Diğer 2 kategori"      # toplanan artık


def test_gercek_diger_kategorisi_artik_dilimle_ayni_renkte_degil():
    """Ölçülmüş hata: dilim rengi etiket metninden çıkarılıyordu ve gerçek `diger`
    kategorisinin adı da "Diğer harcamalar" olduğu için artık dilimle aynı griye
    boyanıyordu — halkada iki gri dilim yan yana."""
    dagilim = [{"kategori": k, "tutar": 100 - i, "pay": 0.1, "islem_adedi": 1}
               for i, k in enumerate(["market", "diger", "akaryakit", "restoran",
                                      "giyim", "seyahat", "telekom"])]

    fig = halka_grafigi(dagilim, 700.0, PALET["light"], "light")
    renkler = dict(zip(fig.data[0].labels, fig.data[0].marker.colors))

    assert renkler["Diğer harcamalar"] == KATEGORI_RENGI["light"]["diger"]
    assert renkler["Diğer 2 kategori"] == ARTIK_RENGI["light"]
    assert renkler["Diğer harcamalar"] != renkler["Diğer 2 kategori"]


def test_hero_kartinda_iki_zaman_olcegi_etiketli():
    """Ölçülmüş karışıklık: "111 işlem · ▼%2 · son 30 gün 8.022 TL" tek satırda
    iki ayrı pencereyi karıştırıyordu; %2 son ayın kıyası, 111 işlem 12 ayın."""
    a = {"toplam_harcama": 85903.0, "islem_adedi": 111, "ortalama_sepet": 774.0,
         "trend": {"son_30_gun": 8022.0, "degisim_orani": -0.02}}

    metin = hero_karti(a, aylar=12, renk=PALET["light"])

    assert "Son 12 ayda harcama" in metin
    assert "12 ayda 111 işlem" in metin
    assert "Son 30 gün" in metin
    assert "önceki iki ayın" in metin


def test_trend_yoksa_hero_karti_kiyas_cumlesi_kurmuyor():
    a = {"toplam_harcama": 100.0, "islem_adedi": 1, "ortalama_sepet": 100.0, "trend": {}}

    metin = hero_karti(a, aylar=3, renk=PALET["light"])

    assert "önceki iki ayın" not in metin
