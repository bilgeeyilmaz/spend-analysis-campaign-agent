"""Araç çıktılarından deterministik Türkçe metin üretir.

Bu modül projenin **LLM'siz yedeğidir**. `GROQ_API_KEY` yoksa, Groq erişilemezse
ya da limit dolarsa sistem buraya düşer ve demo çalışmaya devam eder. Aynı
zamanda `MockProvider`'ın cevap üretme motorudur, yani yedek yol her test
koşusunda kullanılıyor — "hiç denenmemiş fallback" diye bir şey kalmıyor.

ÜSLUP KURALI (tercih değil, uyum gereği): agent **betimler, tavsiye vermez**.
Buradaki cümlelerin hepsi gözlem bildirir — "şu kadar harcadınız", "şu ödeme
tekrarlıyor". Hiçbiri "iptal edin", "tasarruf edin", "yatırım yapın" demez;
yatırım tavsiyesi düzenlemeye tabi bir faaliyettir. Önerilebilecek tek şey
katalogdaki kampanyadır, o da uygunluk kapısından geçmiş olanıdır.
"""

from __future__ import annotations

#: Kategori kodları -> metinde geçecek Türkçe adlar.
KATEGORI_ADI: dict[str, str] = {
    "market": "market",
    "akaryakit": "akaryakıt",
    "restoran": "restoran",
    "giyim": "giyim",
    "elektronik": "elektronik",
    "seyahat": "seyahat",
    "saglik": "sağlık",
    "egitim": "eğitim",
    "eglence": "eğlence",
    "telekom": "telekom",
    "online_alisveris": "online alışveriş",
    "diger": "diğer",
}


def tl(tutar: float) -> str:
    """1234567.8 -> '1.234.568 TL' (Türkçe binlik ayracı)."""
    return f"{tutar:,.0f}".replace(",", ".") + " TL"


def yuzde(oran: float) -> str:
    return f"%{oran * 100:.0f}"


def kategori(kod: str) -> str:
    return KATEGORI_ADI.get(kod, kod)


def harcama_ozeti(analiz: dict, kisa: bool = False) -> str:
    """`analyze_spending` çıktısını paragrafa çevirir.

    Kampanyadan hiç söz etmez — analiz tek başına anlamlı olmak zorunda.

    `kisa=True` öneri metninin içinde kullanılır: orada analiz gerekçedir,
    başlı başına cevap değil. Sistem promptu 5 cümle sınırı koyuyor; uzun özet
    + teklif o sınırı aşıyordu.
    """
    if not analiz.get("islem_var"):
        return "Bu müşteri için incelenecek harcama kaydı bulunamadı."

    limit = 1 if kisa else 2

    cumleler: list[str] = [
        f"Son {analiz['period_months']} ayda {tl(analiz['toplam_harcama'])} tutarında "
        f"{analiz['islem_adedi']} işleminiz var."
    ]

    ilk_uc = analiz.get("kategori_dagilimi", [])[:3]
    if ilk_uc:
        paylar = ", ".join(
            f"{kategori(k['kategori'])} {yuzde(k['pay'])}" for k in ilk_uc
        )
        cumleler.append(f"Harcamanızın dağılımı: {paylar}.")

    trend = analiz.get("trend") or {}
    if not kisa and trend.get("degisim_orani") is not None:
        oran = trend["degisim_orani"]
        yon = "arttı" if oran > 0 else "azaldı"
        if abs(oran) >= 0.15:
            cumleler.append(
                f"Son 30 gündeki harcamanız {tl(trend['son_30_gun'])} — "
                f"önceki aylık ortalamanıza göre {yuzde(abs(oran))} {yon}."
            )

    cumleler.extend(duzenli_gider_cumleleri(analiz.get("duzenli_giderler", []), limit))
    cumleler.extend(artis_cumleleri(analiz.get("olagandisi_artislar", []), limit))
    return " ".join(cumleler)


def duzenli_gider_cumleleri(giderler: list[dict], limit: int = 2) -> list[str]:
    """"Şu işyerine şu kadar süredir aylık şu kadar ödüyorsunuz." — gözlem, öğüt değil."""
    cumleler = []
    for gider in giderler[:limit]:
        cumleler.append(
            f"{kategori(gider['kategori']).capitalize()} kategorisinde "
            f"{gider['ay_sayisi']} aydır aynı üye işyerine ayda ortalama "
            f"{tl(gider['ortalama_tutar'])} ödemeniz görünüyor."
        )
    return cumleler


def artis_cumleleri(artislar: list[dict], limit: int = 2) -> list[str]:
    """İki tespit tipi iki farklı cümle kurar; ayrım `tip` alanından gelir."""
    cumleler = []
    for artis in artislar[:limit]:
        ad = kategori(artis["kategori"])
        if artis.get("tip") == "buyuk_tek_alim":
            taksit = " (taksitli)" if artis.get("taksitli") else ""
            cumleler.append(
                f"{artis['donem'].capitalize()} {ad} kategorisinde "
                f"{tl(artis['tutar'])} tutarında tek bir işleminiz var{taksit}; "
                f"bu kategorideki alışılmış işlem tutarınızın {artis['kat']} katı."
            )
        else:
            cumleler.append(
                f"{ad.capitalize()} harcamanız {artis['donem']} "
                f"{tl(artis['tutar'])} olmuş — önceki dönemlerin ortalaması "
                f"{tl(artis['taban_ortalama'])} idi."
            )
    return cumleler


def kampanya_gerekcesi(analiz: dict, kampanya: dict) -> str:
    """Kampanyayı müşterinin KENDİ verisine bağlar.

    Gerekçe analizden türer; bulunamazsa genel bir cümleye düşer ama uydurma
    bir sebep yazılmaz — "sizin için seçtik" demek gerekçe değildir.
    """
    hedefler = set(kampanya.get("target_categories") or [])

    for artis in analiz.get("olagandisi_artislar", []):
        if artis["kategori"] in hedefler:
            return (
                f"{kategori(artis['kategori'])} kategorisindeki son hareketiniz "
                f"nedeniyle bu kampanya size uygun görünüyor"
            )

    for gider in analiz.get("duzenli_giderler", []):
        if gider["kategori"] in hedefler:
            return (
                f"{kategori(gider['kategori'])} kategorisindeki düzenli ödemeniz "
                f"nedeniyle bu kampanyadan yararlanabilirsiniz"
            )

    for pay in analiz.get("kategori_dagilimi", []):
        if pay["kategori"] in hedefler and pay["pay"] >= 0.05:
            # Ek almadan kuruluyor: Türkçede yüzdeye gelen ek sayının OKUNUŞUNA
            # göre değişiyor (%25'i, %30'u, %16'sı) ve bunu sayıdan türetmek
            # ayrı bir iş. Cümleyi eki gerektirmeyecek şekilde kurmak daha sağlam.
            return (
                f"{kategori(pay['kategori'])} kategorisi harcamanızın "
                f"{yuzde(pay['pay'])} kadarını oluşturuyor"
            )

    return "harcama profilinize göre değerlendirildi"


def kampanya_metni(analiz: dict, oneriler: list[dict], detaylar: dict[str, dict]) -> str:
    """Tam öneri metni: ÖNCE analiz, SONRA teklif.

    Sıra tesadüf değil — projenin konumlandırması bu. Teklifi analizin önüne
    almak sistemi jenerik bir öneri motoruna indirger.
    """
    if not oneriler:
        return (
            harcama_ozeti(analiz)
            + " Şu anda size sunabileceğimiz uygun bir kampanya bulunmuyor."
        )

    en_iyi = oneriler[0]
    detay = detaylar.get(en_iyi["campaign_id"], {})
    gerekce = kampanya_gerekcesi(analiz, detay)

    metin = [harcama_ozeti(analiz, kisa=True)]
    metin.append(
        f"Bu nedenle \"{en_iyi['name']}\" kampanyasını öneriyoruz: {gerekce}."
    )
    if detay.get("description"):
        metin.append(str(detay["description"]))
    if detay.get("min_spend"):
        metin.append(f"Kampanya için asgari harcama tutarı {tl(detay['min_spend'])}.")

    if len(oneriler) > 1:
        digerleri = ", ".join(f"\"{o['name']}\"" for o in oneriler[1:3])
        metin.append(f"Ayrıca {digerleri} kampanyaları da profilinize uygun.")

    return " ".join(metin)
