"""Harcama analizi motoru — PFM anlatısının tamamı buradan çıkar.

Bu modül **kampanya kataloğunu hiç okumaz**. Sebebi mimari bir karar: harcama
analizi projenin birinci sınıf yeteneğidir, teklifin yan ürünü değil. "Son üç
ayınızın %38'i markete gitti" cümlesi, ortada uygun bir kampanya olmasa bile
müşteriye söylenecek anlamlı bir şeydir; öneri bu cümlenin üstüne oturur, tersi
değil. Bu ayrımı bozmak projeyi jenerik bir öneri motoruna indirger.

İçerideki dört analiz:

  kategori kırılımı  — son N ayın kategori payları ve tutarları
  aylık seri         — 12 aylık toplam (mevsimsellik grafiği için)
  düzenli giderler   — aynı üye işyerine tekrarlayan aylık ödemeler
  olağandışı artışlar— son ayda tabanına göre sıçrayan kategoriler

Son ikisi **deterministik dedektörlerdir**, LLM'siz test edilir ve eşikleri
`config.yaml → agent.spending_analysis` altındadır. Eşiklerin kalibrasyonu
üretilen veri taranarak yapıldı; gevşetmenin bedeli config'te yazılı.

Dedektörler üretecin İSİMLENDİRME kurallarına bakmaz. `generator.py` düzenli
ödemelere `M_TELEKOM_S00` gibi ayırt edici bir merchant_id veriyor ve bunu
okumak testi anında yeşile boyardı — ama gerçek veride öyle bir işaret yok.
Tespit yalnızca davranıştan yapılır: aynı işyeri, aylık ritim, sabit tutar.
"""

from __future__ import annotations

import pandas as pd

EPS = 1e-9

#: Aylık ritim sayılması için iki ödeme arasındaki gün farkının düşmesi gereken
#: aralık config'ten gelir; buradaki değerler yalnızca config'siz çağrıda geçerli.
VARSAYILAN = {
    "default_months": 3,
    "series_months": 12,
    "recurring_lookback_months": 6,
    "recurring_min_months": 3,
    "recurring_amount_tolerance": 0.15,
    "recurring_gap_days": (25, 35),
    "anomaly_require_full_baseline": True,
    "anomaly_baseline_months": 3,
    "anomaly_scan_windows": 3,
    "anomaly_single_tx_share": 0.6,
    "outlier_ratio": 6.0,
    "outlier_min_amount": 5000,
    "anomaly_ratio_threshold": 2.5,
    "anomaly_min_baseline": 1500,
    "anomaly_min_amount": 2500,
}


def _pencere(tx: pd.DataFrame, as_of: pd.Timestamp, months: int) -> pd.DataFrame:
    """Son `months` ayın işlemleri (as_of dahil, ay başına hizalı değil).

    Üst sınır şart: `as_of` verideki en son işlem tarihi olduğu sürece fark
    etmiyor, ama `config.features.as_of_date` geçmiş bir tarihe sabitlendiğinde
    (yeniden üretilebilirlik için, ki o alan tam bunun için var) üst sınırsız
    filtre "gelecekteki" işlemleri de bu döneme sayar ve hem tutarları hem
    dönem karşılaştırmasını sessizce şişirir.
    """
    baslangic = as_of - pd.DateOffset(months=months)
    return tx[(tx["transaction_date"] > baslangic) & (tx["transaction_date"] <= as_of)]


def kategori_kirilimi(tx: pd.DataFrame) -> list[dict]:
    """Kategori bazında tutar, pay ve işlem adedi — paya göre azalan."""
    if tx.empty:
        return []
    grup = tx.groupby("mcc_category", observed=True).agg(
        tutar=("amount", "sum"), islem_adedi=("amount", "size")
    )
    toplam = float(grup["tutar"].sum())
    grup["pay"] = grup["tutar"] / (toplam + EPS)
    grup = grup.sort_values("pay", ascending=False)
    return [
        {
            "kategori": str(kategori),
            "tutar": round(float(satir["tutar"]), 2),
            "pay": round(float(satir["pay"]), 4),
            "islem_adedi": int(satir["islem_adedi"]),
        }
        for kategori, satir in grup.iterrows()
    ]


#: Ay numarası -> Türkçe ay adı. Seride hem `ay` ("2026-06") hem `ay_adi`
#: ("Haziran 2026") gönderiliyor: LLM'e yalnız ISO biçimi verildiğinde onu kendi
#: çeviriyor ve cevap "June 2026" diye İngilizce çıkıyordu. Sistem promptuna
#: "Türkçe yaz" yazmak yerine doğru biçimi veriyle vermek daha güvenilir —
#: model çevirmek zorunda kalmıyor.
AY_ADLARI = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]


def ay_adi(donem: str) -> str:
    """'2026-06' -> 'Haziran 2026'."""
    try:
        yil, ay = donem.split("-")
        return f"{AY_ADLARI[int(ay) - 1]} {yil}"
    except (ValueError, IndexError):
        return donem


def aylik_seri(tx: pd.DataFrame, as_of: pd.Timestamp, months: int) -> list[dict]:
    """Aylık toplam harcama serisi. Harcaması olmayan ay 0 olarak yer alır.

    Boş ayı atlamak yerine sıfırla doldurmak bilinçli: grafikte kesinti yerine
    düşüş görünür ve "geçen ay hiç harcamamışsınız" ifadesi mümkün olur.
    """
    pencere = _pencere(tx, as_of, months)
    aylar = pd.period_range(
        end=as_of.to_period("M"), periods=months, freq="M"
    )
    if pencere.empty:
        return [{"ay": str(ay), "ay_adi": ay_adi(str(ay)), "tutar": 0.0,
                 "islem_adedi": 0} for ay in aylar]

    grup = pencere.groupby(pencere["transaction_date"].dt.to_period("M")).agg(
        tutar=("amount", "sum"), islem_adedi=("amount", "size")
    ).reindex(aylar, fill_value=0)

    return [
        {"ay": str(ay), "ay_adi": ay_adi(str(ay)),
         "tutar": round(float(satir["tutar"]), 2),
         "islem_adedi": int(satir["islem_adedi"])}
        for ay, satir in grup.iterrows()
    ]


def duzenli_giderler(tx: pd.DataFrame, as_of: pd.Timestamp, ayar: dict) -> list[dict]:
    """Tekrarlayan aylık ödemeleri tespit eder.

    Üç şart birden aranır — üçü de gerçek veride hesaplanabilir olsun diye
    yalnız işlem alanlarından türetiliyor:

      1. aynı `merchant_id`'de en az `recurring_min_months` FARKLI ayda ödeme,
      2. tutar oynaması `recurring_amount_tolerance` içinde
         (maks-min farkı ortalamanın bu oranını aşmayacak),
      3. ardışık ödemeler arası gün farkının ORTANCASI `recurring_gap_days`
         aralığında — yani ritim aylık.

    Ortanca kullanılıyor çünkü tek bir kaçan ödeme (60 günlük boşluk) ortalamayı
    aralık dışına atıp gerçek bir aboneliği gizler.
    """
    lookback = ayar.get("recurring_lookback_months", VARSAYILAN["recurring_lookback_months"])
    min_ay = ayar.get("recurring_min_months", VARSAYILAN["recurring_min_months"])
    tolerans = ayar.get("recurring_amount_tolerance", VARSAYILAN["recurring_amount_tolerance"])
    alt, ust = ayar.get("recurring_gap_days", VARSAYILAN["recurring_gap_days"])

    pencere = _pencere(tx, as_of, lookback)
    if pencere.empty:
        return []

    bulunanlar: list[dict] = []
    for (merchant, kategori), grup in pencere.groupby(
        ["merchant_id", "mcc_category"], observed=True
    ):
        tarihler = grup["transaction_date"].sort_values()
        if tarihler.dt.to_period("M").nunique() < min_ay:
            continue

        tutarlar = grup["amount"]
        ortalama = float(tutarlar.mean())
        if ortalama <= 0:
            continue
        oynama = float(tutarlar.max() - tutarlar.min()) / ortalama
        if oynama > tolerans:
            continue

        farklar = tarihler.diff().dropna().dt.days
        if farklar.empty or not (alt <= float(farklar.median()) <= ust):
            continue

        bulunanlar.append({
            "merchant_id": str(merchant),
            "kategori": str(kategori),
            "ortalama_tutar": round(ortalama, 2),
            "ay_sayisi": int(tarihler.dt.to_period("M").nunique()),
            "tipik_gun": int(tarihler.dt.day.median()),
            "son_odeme": tarihler.max().date().isoformat(),
            "aylik_toplam_etki": round(ortalama, 2),
        })

    return sorted(bulunanlar, key=lambda r: r["ortalama_tutar"], reverse=True)


#: Pencere indeksi -> insan diline çevrilmiş dönem adı (agent metni için).
DONEM_ADI = {0: "son 30 gün", 1: "31-60 gün önce", 2: "61-90 gün önce"}


def olagandisi_artislar(tx: pd.DataFrame, as_of: pd.Timestamp, ayar: dict) -> list[dict]:
    """Son 3 ayda tabanına göre sıçrayan kategoriler.

    NEDEN TAKVİM AYI DEĞİL, 30 GÜNLÜK KAYAN PENCERE: `as_of` ayın ortasına
    denk geldiğinde (veride 13 Ağustos) "son ay" yarım aydır ve tam aylardan
    oluşan bir tabanla kıyaslanınca oran sistematik olarak bastırılır — gerçek
    sıçramalar sessizce kaybolur. Kayan pencerede her dönem eşit uzunlukta.

    NEDEN ÜÇ PENCERE BİRDEN: üreteç sıçramayı son ÜÇ aydan birine ekliyor.
    Yalnız en son pencereye bakmak sıçramaların üçte ikisini kaçırır ve dahası
    onları TABANA yazarak tespiti bir kat daha zorlaştırır.

    Her aday pencere kendinden önceki `anomaly_baseline_months` pencereye karşı
    sınanır ve DÖRT şart birden aranır:

      1. taban pencerelerinin HEPSİNDE o kategoride harcama var (taban istikrarlı),
      2. taban ortalaması >= `anomaly_min_baseline`,
      3. aday pencere >= taban * `anomaly_ratio_threshold`,
      4. mutlak fark >= `anomaly_min_amount` (söylemeye değer bir tutar).

    Dördü birden aranmasının sebebi config'te yazılı: tek başına "2.5 kat"
    kuralı müşterilerin %96'sını işaretliyordu ve %96 uyarı vermek hiç uyarı
    vermemekle aynı şeydir.
    """
    taban_n = ayar.get("anomaly_baseline_months", VARSAYILAN["anomaly_baseline_months"])
    tam_taban = ayar.get("anomaly_require_full_baseline", VARSAYILAN["anomaly_require_full_baseline"])
    kat = ayar.get("anomaly_ratio_threshold", VARSAYILAN["anomaly_ratio_threshold"])
    min_taban = ayar.get("anomaly_min_baseline", VARSAYILAN["anomaly_min_baseline"])
    min_fark = ayar.get("anomaly_min_amount", VARSAYILAN["anomaly_min_amount"])
    tarama = ayar.get("anomaly_scan_windows", VARSAYILAN["anomaly_scan_windows"])
    tek_islem_esigi = ayar.get("anomaly_single_tx_share", VARSAYILAN["anomaly_single_tx_share"])

    if tx.empty:
        return []

    # Pencere 0 = son 30 gün, pencere 1 = 31-60 gün önce, ...
    toplam_pencere = tarama + taban_n
    gunler = (as_of - tx["transaction_date"]).dt.days
    pencere_no = gunler // 30
    ilgili = tx[(gunler >= 0) & (pencere_no < toplam_pencere)]
    if ilgili.empty:
        return []

    pivot = ilgili.pivot_table(
        index=pencere_no[ilgili.index], columns="mcc_category",
        values="amount", aggfunc="sum", observed=True,
    ).reindex(range(toplam_pencere), fill_value=0.0).fillna(0.0)
    # En büyük tek işlem: artışın tek bir alımdan mı yoksa yayılmış bir
    # harcama artışından mı geldiğini ayırt etmek için.
    pivot_max = ilgili.pivot_table(
        index=pencere_no[ilgili.index], columns="mcc_category",
        values="amount", aggfunc="max", observed=True,
    ).reindex(range(toplam_pencere), fill_value=0.0).fillna(0.0)

    bulunanlar: list[dict] = []
    for aday in range(tarama):
        taban_pencereleri = list(range(aday + 1, aday + 1 + taban_n))
        for kategori in pivot.columns:
            taban_seri = pivot.loc[taban_pencereleri, kategori]
            deger = float(pivot.loc[aday, kategori])

            if tam_taban and (taban_seri <= 0).any():
                continue
            taban = float(taban_seri.mean())
            if taban < min_taban or deger < taban * kat or deger - taban < min_fark:
                continue

            tek_islem_payi = float(pivot_max.loc[aday, kategori]) / (deger + EPS)
            bulunanlar.append({
                "tip": "kategori_artisi",
                "kategori": str(kategori),
                "donem": DONEM_ADI.get(aday, f"{aday*30+1}-{(aday+1)*30} gün önce"),
                "donem_index": aday,
                "tutar": round(deger, 2),
                "taban_ortalama": round(taban, 2),
                "kat": round(deger / (taban + EPS), 2),
                "fark": round(deger - taban, 2),
                "tek_islemden_mi": bool(tek_islem_payi >= tek_islem_esigi),
            })

    # Aynı kategori birden çok pencerede işaretlenebilir; en yenisi kalsın.
    en_yeni: dict[str, dict] = {}
    for kayit in bulunanlar:
        onceki = en_yeni.get(kayit["kategori"])
        if onceki is None or kayit["donem_index"] < onceki["donem_index"]:
            en_yeni[kayit["kategori"]] = kayit

    return sorted(en_yeni.values(), key=lambda r: r["fark"], reverse=True)


def buyuk_alimlar(tx: pd.DataFrame, as_of: pd.Timestamp, ayar: dict) -> list[dict]:
    """Müşterinin kendi alışkanlığına göre olağandışı büyük TEK işlemler.

    `olagandisi_artislar` kategori TOPLAMINA bakar ve istikrarlı bir taban ister;
    bu, tek seferlik büyük alımların çoğunu kaçırır — insanlar büyük alımı
    genelde düzenli harcamadıkları bir kategoride yapar (12 bin TL'lik seyahat,
    45 bin TL'lik elektronik) ve orada taban yoktur. Ölçüldü: taban şartı tek
    başına yakalanabilir sıçramaları %13'e indiriyordu.

    Bu dedektör tabana değil, MÜŞTERİNİN KENDİ SEPET MEDYANINA bakar: o
    kategoride tipik olarak 400 TL harcayan birinin 9.000 TL'lik işlemi
    olağandışıdır, aynı işlem private segment bir müşteri için olmayabilir.
    Eşik mutlak değil görecelidir; "5.000 TL üstü her işlem" kuralı zengin
    müşteriyi sürekli, dar bütçeliyi hiç işaretlerdi.

    Ölçüm (2.000 müşteri, üretecin ektiği sıçramalara karşı): precision %42,
    recall %45, **lift 3.0x**. Kategori toplamına bakan dedektörün lift'i
    1.13x — yani tesadüften farksızdı; ayrımın tamamı bu fonksiyondan geliyor.
    """
    oran_esigi = ayar.get("outlier_ratio", VARSAYILAN["outlier_ratio"])
    min_tutar = ayar.get("outlier_min_amount", VARSAYILAN["outlier_min_amount"])
    tarama = ayar.get("anomaly_scan_windows", VARSAYILAN["anomaly_scan_windows"])

    if tx.empty:
        return []

    gun = (as_of - tx["transaction_date"]).dt.days
    pencere_gun = tarama * 30
    aday = tx[(gun >= 0) & (gun < pencere_gun)]
    gecmis = tx[gun >= pencere_gun]
    if aday.empty:
        return []

    # Tipik sepet GEÇMİŞTEN hesaplanır: aday işlemi kendi referansına katmak,
    # tek işlemi olan bir kategoride oranı her zaman 1.0 yapıp tespiti öldürür.
    kategori_medyan = gecmis.groupby("mcc_category", observed=True)["amount"].median()
    genel_medyan = float(gecmis["amount"].median()) if not gecmis.empty else float(tx["amount"].median())

    bulunanlar: list[dict] = []
    for _, satir in aday.iterrows():
        tutar = float(satir["amount"])
        if tutar < min_tutar:
            continue
        tipik = float(kategori_medyan.get(satir["mcc_category"], genel_medyan))
        if not tipik > 0:
            tipik = genel_medyan
        if not tipik > 0 or tutar < tipik * oran_esigi:
            continue

        gecen_gun = int((as_of - satir["transaction_date"]).days)
        bulunanlar.append({
            "tip": "buyuk_tek_alim",
            "kategori": str(satir["mcc_category"]),
            "tutar": round(tutar, 2),
            "tipik_sepet": round(tipik, 2),
            "kat": round(tutar / tipik, 1),
            "tarih": satir["transaction_date"].date().isoformat(),
            "donem": DONEM_ADI.get(gecen_gun // 30, f"{gecen_gun} gün önce"),
            "donem_index": gecen_gun // 30,
            "taksitli": int(satir.get("installment_count", 1)) > 1,
        })

    # Kategori başına en büyüğü yeter; agent'a beş satırlık liste vermenin anlamı yok.
    en_buyuk: dict[str, dict] = {}
    for kayit in bulunanlar:
        onceki = en_buyuk.get(kayit["kategori"])
        if onceki is None or kayit["tutar"] > onceki["tutar"]:
            en_buyuk[kayit["kategori"]] = kayit
    return sorted(en_buyuk.values(), key=lambda r: r["tutar"], reverse=True)


def analiz_et(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    months: int | None = None,
    ayar: dict | None = None,
) -> dict:
    """Tek müşterinin işlemlerinden tam harcama analizini üretir.

    `transactions` TEK müşteriye ait olmalıdır; filtreleme çağıranın işidir
    (gerçek veride bu bir WHERE'e döner, bkz. `DataSource.get_transactions`).
    """
    ayar = {**VARSAYILAN, **(ayar or {})}
    months = months or ayar["default_months"]

    if transactions.empty:
        return {
            "as_of": None, "period_months": months, "islem_var": False,
            "toplam_harcama": 0.0, "islem_adedi": 0, "ortalama_sepet": 0.0,
            "kategori_dagilimi": [], "ilk_uc_kategori": [], "aylik_seri": [],
            "trend": None, "duzenli_giderler": [], "olagandisi_artislar": [],
            "kategori_karsilastirma": [], "ilk_uye_isyerleri": [],
        }

    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    as_of = pd.Timestamp(as_of) if as_of is not None else tx["transaction_date"].max()

    pencere = _pencere(tx, as_of, months)
    kirilim = kategori_kirilimi(pencere)
    toplam = float(pencere["amount"].sum())

    return {
        "as_of": as_of.date().isoformat(),
        "period_months": months,
        "islem_var": True,
        "toplam_harcama": round(toplam, 2),
        "islem_adedi": int(len(pencere)),
        "ortalama_sepet": round(toplam / max(len(pencere), 1), 2),
        "kategori_dagilimi": kirilim,
        "ilk_uc_kategori": [k["kategori"] for k in kirilim[:3]],
        "aylik_seri": aylik_seri(tx, as_of, ayar["series_months"]),
        "trend": _trend(tx, as_of),
        "kategori_karsilastirma": kategori_karsilastirma(tx, as_of, months),
        "ilk_uye_isyerleri": ilk_uye_isyerleri(tx, as_of, months),
        "duzenli_giderler": duzenli_giderler(tx, as_of, ayar),
        # İki tetikleme yolu tek listede toplanıyor, `tip` alanı hangisi olduğunu
        # söylüyor. Agent'ın cümlesi buna göre değişir: "elektronikte 45.000 TL'lik
        # tek bir alım" ile "markette genel bir artış" aynı şey değil.
        "olagandisi_artislar": (
            buyuk_alimlar(tx, as_of, ayar) + olagandisi_artislar(tx, as_of, ayar)
        ),
    }


def kategori_karsilastirma(
    tx: pd.DataFrame, as_of: pd.Timestamp, months: int
) -> list[dict]:
    """Bu dönemin kategori harcamalarını **eşit uzunlukta önceki dönemle** kıyaslar.

    Karşılaştırma penceresi bilerek "önceki ay" değil "önceki eşit dönem": kullanıcı
    3 aylık pencereye baktığında onu 1 aylık bir dönemle kıyaslamak farkı üçe
    katlanmış gösterir. Kategoriler iki dönemin BİRLEŞİMİDİR — bu dönem hiç
    harcanmamış bir kategorideki düşüş de görünmeli, yoksa "market harcamam
    bitti" bilgisi sessizce kaybolur.
    """
    bu_donem = _pencere(tx, as_of, months)
    onceki_sinir = as_of - pd.DateOffset(months=months)
    onceki_donem = tx[
        (tx["transaction_date"] <= onceki_sinir)
        & (tx["transaction_date"] > onceki_sinir - pd.DateOffset(months=months))
    ]

    simdi = bu_donem.groupby("mcc_category", observed=True)["amount"].sum()
    once = onceki_donem.groupby("mcc_category", observed=True)["amount"].sum()

    satirlar = []
    for kategori in sorted(set(simdi.index) | set(once.index)):
        a = float(simdi.get(kategori, 0.0))
        b = float(once.get(kategori, 0.0))
        satirlar.append({
            "kategori": kategori,
            "tutar": round(a, 2),
            "onceki_tutar": round(b, 2),
            "fark": round(a - b, 2),
            # Önceki dönem sıfırsa oran tanımsız: "sonsuz artış" yazmak yerine
            # None dönüp arayüzün "yeni" demesine izin veriyoruz.
            "degisim_orani": round(a / b - 1, 4) if b > 0 else None,
        })
    return sorted(satirlar, key=lambda d: d["tutar"], reverse=True)


def ilk_uye_isyerleri(tx: pd.DataFrame, as_of: pd.Timestamp, months: int,
                      n: int = 8) -> list[dict]:
    """Dönemin en çok harcanan üye işyerleri.

    Kategori "neye" harcadığını söyler, üye işyeri "nereye" — ikisi farklı
    sorulardır ve kullanıcı ikincisini de sorar.
    """
    pencere = _pencere(tx, as_of, months)
    if pencere.empty:
        return []

    grup = pencere.groupby("merchant_id").agg(
        tutar=("amount", "sum"),
        islem_adedi=("amount", "size"),
        kategori=("mcc_category", lambda s: s.mode().iat[0]),
    ).sort_values("tutar", ascending=False).head(n)

    return [
        {"merchant_id": str(m), "kategori": str(r.kategori),
         "tutar": round(float(r.tutar), 2), "islem_adedi": int(r.islem_adedi)}
        for m, r in grup.iterrows()
    ]


def _trend(tx: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Son 30 gün ile önceki 60 günün aylık ortalaması karşılaştırması.

    30'a 30 karşılaştırmak tek bir büyük alışverişte %200 "artış" üretiyordu;
    önceki 60 günün AYLIK ORTALAMASI referans alınınca gürültü yarıya iniyor.
    """
    son = tx[tx["transaction_date"] > as_of - pd.Timedelta(days=30)]["amount"].sum()
    onceki = tx[
        (tx["transaction_date"] <= as_of - pd.Timedelta(days=30))
        & (tx["transaction_date"] > as_of - pd.Timedelta(days=90))
    ]["amount"].sum() / 2.0

    return {
        "son_30_gun": round(float(son), 2),
        "onceki_aylik_ortalama": round(float(onceki), 2),
        "degisim_orani": round(float(son / onceki - 1), 4) if onceki > 0 else None,
    }
