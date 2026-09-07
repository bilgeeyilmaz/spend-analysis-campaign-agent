"""Segment isimlendirme ve küme sayısı seçimi testleri.

`tests/test_features.py` segmentasyonu uçtan uca (gerçek özellik tablosuyla)
sınıyor: her müşteri bir segmente düştü mü, mikro küme var mı, isimler benzersiz
mi. Burada sınanan şey farklı — **karar kurallarının kendisi**:

  1. `isimlendir()` bir kategoriyi isme neden alır? Cevap "payı büyük olduğu
     için" değil, "nüfusa göre payı yüksek olduğu için" (lift) olmalı. Bu ayrım
     gerçek veride sessizce bozulabilecek türden: market payı herkeste yüksektir,
     mutlak paya bakan bir isimlendirme her segmenti "market odaklı" yapar.
  2. `en_iyi_k()` en yüksek silhouette'i mi seçer? Hayır — önce iş kısıtını
     uygular. Kısıt kalkarsa hiçbir test kırmızıya dönmüyordu.

Testler kendi girdisini elde kurar: rastgele veri üretip "acaba ne çıkacak"
diye bakmak yerine, kuralın cevabının ne olması gerektiği baştan bellidir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.categories import CATEGORIES
from src.models.segmentation import (
    KATEGORI_ETIKET,
    LIFT_ESIK,
    MIN_KUME_ORANI,
    en_iyi_k,
    isimlendir,
)

RASTGELE = 42


# --------------------------------------------------------------------------
# Yardımcılar: isimlendirme girdisini elde kurmak
# --------------------------------------------------------------------------


def _profil(*kumeler: dict[str, float]) -> pd.DataFrame:
    """Küme × özellik profil tablosu. Verilmeyen kategori payı 0 sayılır."""
    satirlar = []
    for kume in kumeler:
        satir = {f"cat_share_{c}": 0.0 for c in CATEGORIES}
        satir["monetary_total"] = kume.get("monetary_total", 100_000.0)
        for anahtar, deger in kume.items():
            if anahtar != "monetary_total":
                satir[f"cat_share_{anahtar}"] = deger
        satirlar.append(satir)
    return pd.DataFrame(satirlar, index=range(len(satirlar)))


def _nufus(**paylar: float) -> pd.Series:
    """Nüfus ortalaması. Profil ile aynı kolon setini taşımak zorunda."""
    seri = {f"cat_share_{c}": 0.0 for c in CATEGORIES}
    seri["monetary_total"] = 100_000.0
    for anahtar, deger in paylar.items():
        seri[f"cat_share_{anahtar}"] = deger
    return pd.Series(seri)


#: Nüfusun harcama dağılımı; tercilleri ~400k ve ~700k'ya denk gelir.
HARCAMA = pd.Series(np.linspace(100_000, 1_000_000, 300))


# --------------------------------------------------------------------------
# isimlendir(): kategori seçimi lift ile yapılır
# --------------------------------------------------------------------------


def test_lift_esigi_altindaki_kategori_isme_girmez():
    """Nüfusla aynı profildeki küme ayırt edici değildir: "dengeli harcayan"."""
    profil = _profil({"market": 0.30, "restoran": 0.20, "seyahat": 0.05})
    nufus = _nufus(market=0.30, restoran=0.20, seyahat=0.05)

    ad = isimlendir(profil, nufus, HARCAMA)[0]

    assert "dengeli harcayan" in ad
    assert "market" not in ad


def test_yuksek_liftli_kategori_isme_girer():
    profil = _profil({"seyahat": 0.24, "market": 0.20})
    nufus = _nufus(seyahat=0.10, market=0.20)      # seyahat lift 2.4

    ad = isimlendir(profil, nufus, HARCAMA)[0]

    assert KATEGORI_ETIKET["seyahat"] in ad


def test_mutlak_pay_degil_lift_belirler():
    """İsimlendirmenin can alıcı noktası.

    Kümenin en büyük payı market (%30) ama bu nüfus ortalamasının aynısı —
    yani hiçbir şey söylemiyor. Eğitim payı küçük (%12) ama nüfusun 3 katı.
    İsme eğitim girmeli. Mutlak paya dönülürse bu test kırılır.
    """
    profil = _profil({"market": 0.30, "egitim": 0.12})
    nufus = _nufus(market=0.30, egitim=0.04)

    ad = isimlendir(profil, nufus, HARCAMA)[0]

    assert KATEGORI_ETIKET["egitim"] in ad
    assert KATEGORI_ETIKET["market"] not in ad


def test_lift_esigi_gercekten_esik_gibi_davraniyor():
    """Eşiğin hemen altı elenir, hemen üstü geçer."""
    altinda = _profil({"giyim": 0.10 * (LIFT_ESIK - 0.10)})
    ustunde = _profil({"giyim": 0.10 * (LIFT_ESIK + 0.10)})
    nufus = _nufus(giyim=0.10)

    assert KATEGORI_ETIKET["giyim"] not in isimlendir(altinda, nufus, HARCAMA)[0]
    assert KATEGORI_ETIKET["giyim"] in isimlendir(ustunde, nufus, HARCAMA)[0]


def test_nufusta_hic_gorulmeyen_kategori_ismi_bozmaz():
    """Nüfus payı 0 olan kategoride lift tanımsızdır (0'a bölme).

    Gerçek veride bir kategori hiç görülmeyebilir; bu durumda isimlendirme
    patlamamalı ve tanımsız lift'i "sonsuz derecede ayırt edici" sayıp ismi
    ele geçirmemeli.
    """
    profil = _profil({"saglik": 0.02, "seyahat": 0.24})
    nufus = _nufus(saglik=0.0, seyahat=0.10)

    ad = isimlendir(profil, nufus, HARCAMA)[0]

    assert KATEGORI_ETIKET["seyahat"] in ad
    assert KATEGORI_ETIKET["saglik"] not in ad


def test_kategori_etiket_sozlugu_tum_kategorileri_kapsiyor():
    """CATEGORIES'e yeni kategori eklenip etiketi unutulursa isimlendirme
    KeyError ile düşer — hata modelin ortasında değil, burada görünsün."""
    eksik = sorted(set(CATEGORIES) - set(KATEGORI_ETIKET))
    assert not eksik, f"KATEGORI_ETIKET'te karşılığı olmayan kategori: {eksik}"


# --------------------------------------------------------------------------
# isimlendir(): harcama düzeyi nüfus tercillerinden okunur
# --------------------------------------------------------------------------


def test_harcama_duzeyi_nufus_tercilinden_okunur():
    """Kümelerin kendi içindeki tercili kullanılırsa 180 bin TL'lik küme de
    "yüksek harcamalı" olur (kendi üçlüsünün en üstü olduğu için). Ölçek nüfus
    olmalı: 180k tutumlu, 670k orta, 950k yüksek."""
    profil = _profil(
        {"monetary_total": 180_000.0},
        {"monetary_total": 670_000.0},
        {"monetary_total": 950_000.0},
    )
    nufus = _nufus()

    isimler = isimlendir(profil, nufus, HARCAMA)

    assert isimler[0].startswith("tutumlu")
    assert isimler[1].startswith("orta segment")
    assert isimler[2].startswith("yüksek harcamalı")


def test_tum_kumeler_zenginse_hepsi_yuksek_olabilir():
    """Nüfus ölçeğinin doğal sonucu: üç küme de üst tercildeyse üçü de yüksek
    harcamalıdır. Küme içi tercil bunu yapay olarak dağıtırdı."""
    profil = _profil(
        {"monetary_total": 800_000.0, "seyahat": 0.24},
        {"monetary_total": 900_000.0, "giyim": 0.24},
        {"monetary_total": 990_000.0, "elektronik": 0.24},
    )
    nufus = _nufus(seyahat=0.10, giyim=0.10, elektronik=0.10)

    isimler = isimlendir(profil, nufus, HARCAMA)

    assert all(ad.startswith("yüksek harcamalı") for ad in isimler.values())


# --------------------------------------------------------------------------
# isimlendir(): isimler benzersiz kalmalı
# --------------------------------------------------------------------------


def test_ayni_baskin_kategoride_isimler_ayrisir():
    """İki küme aynı düzey + aynı baskın kategoriye düşerse isim ikinci
    kategoriyle ayrıştırılır; demoda iki segment aynı etiketi taşıyamaz."""
    profil = _profil(
        {"monetary_total": 950_000.0, "seyahat": 0.24, "giyim": 0.05},
        {"monetary_total": 960_000.0, "seyahat": 0.24, "giyim": 0.24},
    )
    nufus = _nufus(seyahat=0.10, giyim=0.10)

    isimler = isimlendir(profil, nufus, HARCAMA)

    assert len(set(isimler.values())) == 2
    assert KATEGORI_ETIKET["giyim"] in isimler[1]


def test_hicbir_ayirt_edici_kategori_yoksa_bile_isimler_benzersiz():
    """En kötü hâl: iki küme de nüfusun aynısı ve aynı harcama düzeyinde.
    Ayrıştıracak kategori yok — isim yine de tekrar etmemeli."""
    profil = _profil(
        {"monetary_total": 200_000.0, "market": 0.30},
        {"monetary_total": 210_000.0, "market": 0.30},
    )
    nufus = _nufus(market=0.30)

    isimler = isimlendir(profil, nufus, HARCAMA)

    assert len(set(isimler.values())) == 2


# --------------------------------------------------------------------------
# en_iyi_k(): önce iş kısıtı, sonra silhouette
# --------------------------------------------------------------------------


def _kumeli_veri(boyutlar: list[int], mesafe: float = 50.0) -> np.ndarray:
    """Verilen büyüklüklerde, birbirinden uzak ve kendi içinde sıkı kümeler."""
    rng = np.random.default_rng(RASTGELE)
    parcalar = [
        rng.normal(loc=[i * mesafe, i * mesafe], scale=0.5, size=(n, 2))
        for i, n in enumerate(boyutlar)
    ]
    return np.vstack(parcalar)


def test_ayrik_kumeler_dogru_k_ile_bulunur():
    """Kısıt devrede değilken beklenen davranış: 3 ayrık küme -> k=3."""
    X = _kumeli_veri([80, 80, 80])

    k, skorlar, _ = en_iyi_k(X, [2, 3, 4], RASTGELE, ornek=1000)

    assert k == 3
    assert skorlar[3] == max(skorlar.values())


def test_mikro_kume_ureten_k_silhouette_yuksek_olsa_bile_elenir():
    """Testin bütün anlamı burada: elenen k, en yüksek skoru alan k olmalı.

    İki büyük küme + 3 kişilik uzak bir aykırı grup. k=3 bu yapıyı mükemmel
    ayırır ve en yüksek silhouette'i alır, ama üçüncü küme nüfusun %1,5'i —
    kampanya kurgulanamayacak bir "artık". Beklenen sonuç k=2.
    """
    X = _kumeli_veri([100, 100, 3])

    k, skorlar, en_kucuk = en_iyi_k(X, [2, 3], RASTGELE, ornek=1000)

    assert skorlar[3] > skorlar[2], "kurgu bozulmuş: elenen k zaten kaybediyordu"
    assert en_kucuk[3] < MIN_KUME_ORANI <= en_kucuk[2]
    assert k == 2, "mikro küme üreten k seçildi — iş kısıtı devre dışı kalmış"


def test_hicbir_k_kisiti_saglamazsa_hata_verir():
    """Sessizce kötü bir k dönmektense patlamak doğru: çıktı sunumda
    "modelin bulduğu segmentler" diye gösterilecek."""
    X = np.vstack([_kumeli_veri([200]), [[5_000.0, 5_000.0]], [[-5_000.0, -5_000.0]]])

    with pytest.raises(ValueError, match="MIN_KUME_ORANI|en küçük küme"):
        en_iyi_k(X, [2, 3], RASTGELE, ornek=1000)


def test_raporlanan_skorlar_tum_k_degerlerini_kapsar():
    """Elenen k'lar da raporlanır — `main()` "k=5 elendi: en küçük küme %1.1"
    satırını bu sözlüklerden basıyor, karar şeffaf kalsın diye."""
    k_range = [2, 3, 4]
    X = _kumeli_veri([80, 80, 80])

    _, skorlar, en_kucuk = en_iyi_k(X, k_range, RASTGELE, ornek=1000)

    assert sorted(skorlar) == k_range
    assert sorted(en_kucuk) == k_range
    assert all(0.0 < oran <= 1.0 for oran in en_kucuk.values())


def test_ayni_tohumla_ayni_k():
    """random_state sabitken sonuç yeniden üretilebilir olmalı."""
    X = _kumeli_veri([80, 80, 80])

    ilk = en_iyi_k(X, [2, 3, 4], RASTGELE, ornek=1000)
    ikinci = en_iyi_k(X, [2, 3, 4], RASTGELE, ornek=1000)

    assert ilk[0] == ikinci[0]
    assert ilk[1] == ikinci[1]
