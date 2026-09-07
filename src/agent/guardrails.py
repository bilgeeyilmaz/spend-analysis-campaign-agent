"""LLM çıktısının araç verisine karşı denetimi.

    from src.agent.guardrails import denetle
    sonuc = denetle(cevap_metni, tool_calls, katalog)

TEMEL FİKİR: bu projede doğru cevabı zaten biliyoruz. Araçlar rakamları
hesaplıyor, LLM'in işi onları Türkçeye çevirmek. Dolayısıyla metindeki her sayı
bir araç çıktısına dayanmak zorundadır; dayanmıyorsa uydurmadır. Bu, "LLM'e
güven ama doğrula" tavsiyesinin ölçülebilir hâlidir.

ÜÇ DENETİM, ÜÇ FARKLI HATA TÜRÜ:

  izlenebilirlik  UYDURMA sayı. gpt-oss-20b veride hiç olmayan "9.215 TL'lik
                  giyim aboneliği" üretti. Metindeki sayı araç çıktılarının
                  hiçbirinde yoksa yakalanır.

  odul            ATIF hatası. gpt-oss-120b "%25 nakit iade" yazdı; kampanya %8
                  veriyor, 25 ise müşterinin market payıydı. Sayı veride VAR,
                  ama yanlış yere bağlanmış — izlenebilirlik denetimi bunu
                  göremez, çünkü 25 gerçekten araç çıktısında geçiyor. Bu yüzden
                  ödül oranı ayrıca ve hedefli olarak kontrol edilir.

  tavsiye         Düzenleme ihlali. "İptal edin", "yatırım yapın" gibi ifadeler
                  yatırım tavsiyesi sayılır ve bu sistemin yetkisi dışındadır.

Denetim SONUCU ÜRETİR, karar vermez. Cevabı reddetmek ya da deterministik metne
düşmek orkestratörün işi — burada politika değil ölçüm var.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from src.data.validate import PII_COLUMNS

#: Türkçe metinden sayı yakalar: "53.609,49" | "1.191" | "7,6" | "8"
#: Binlik ayracı nokta VEYA boşluk olabilir (LLM ikisini de üretiyor).
#: Ondalık ayracı tek sınıfta ([.,]) toplandı: ayrı alternatifler yazılırsa
#: "7.6" için önce `\d+` denenip "7"de duruyordu.
SAYI_DESENI = re.compile(
    r"(?<![\w,.])(\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])"
)

#: Sayının hemen öncesinde yüzde işareti var mı — oran/yüzde eşlemesi için.
YUZDE_ONEKI = re.compile(r"%\s*$")

#: Yasak yönlendirme kalıpları. Kök hâlinde tutuluyor ki çekimleri de yakalansın.
TAVSIYE_KALIPLARI: tuple[str, ...] = (
    "iptal ed", "tasarruf", "yatırım yap", "yatirim yap", "biriktir",
    "harcamanızı azalt", "harcamanizi azalt", "kısıtla", "kisitla",
    "borçlan", "borclan", "kredi çek", "kredi cek",
)

#: Bağlamdan bağımsız, her metinde geçebilecek küçük sayılar. Bunları
#: izlenebilirlik denetiminden muaf tutuyoruz — "3 ay", "ilk 3 kategori" gibi
#: ifadeler yüzünden sürekli yanlış alarm üretmesin.
MUAF_SAYILAR: frozenset[float] = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 30.0, 90.0})

#: Eşleşme toleransı DAR tutuluyor. Gerçek veride araç çıktısı yüzlerce sayı
#: içeriyor (12 aylık seri × 12 kategori × 3 metrik); geniş tolerans bu yoğun
#: havuzla birlikte sayı ekseninin çoğunu kapsayıp denetimi boşa çıkarıyordu —
#: ölçüldü: %2 bağıl toleransta uydurulmuş tutarların yalnızca %70'i, %0,1'de
#: %100'ü yakalanıyor. Yuvarlama farkları tolerans yerine aşağıdaki
#: `_yuvarlama_varyantlari` ile karşılanıyor.
#:
#: Asimetri bilinçli: yanlış alarmın bedeli deterministik metne düşmek (metin
#: yine doğru), kaçırmanın bedeli müşteriye yanlış rakam gitmesi.
BAGIL_TOLERANS = 0.001
MUTLAK_TOLERANS = 0.2

#: Bir sayının "oran" sayılıp yüzdeye çevrilebileceği üst sınır. 10 = %1000'e
#: kadar artış. Daha yükseği tutar/adet olur, oran değil.
ORAN_UST_SINIRI = 10.0


@dataclass
class DenetimSonucu:
    gecti: bool = True
    ihlaller: list[str] = field(default_factory=list)
    izlenemeyen_sayilar: list[float] = field(default_factory=list)

    def ekle(self, ihlal: str) -> None:
        self.ihlaller.append(ihlal)
        self.gecti = False


# --------------------------------------------------------------------------
# Sayı çıkarma ve normalleştirme
# --------------------------------------------------------------------------


def _normalize(ham: str) -> float | None:
    """'53.609,49' -> 53609.49 | '7,6' -> 7.6 | '7.6' -> 7.6

    Nokta yalnızca ARDINDAN TAM 3 RAKAM geliyorsa binlik ayracıdır. Bu ayrım
    şart: "7.6 katı" ifadesinde nokta ondalıktır, "3.651 TL"de binliktir.
    """
    metin = ham.replace(" ", " ").strip()
    metin = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", metin)         # her tür boşluklu binlik
    metin = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", metin)           # noktalı binlik
    metin = metin.replace(",", ".")
    try:
        return float(metin)
    except ValueError:
        return None


def sayilari_cikar(metin: str) -> list[tuple[float, bool]]:
    """Metindeki sayıları (değer, yüzde_mi) olarak döner."""
    bulunanlar: list[tuple[float, bool]] = []
    for eslesme in SAYI_DESENI.finditer(metin):
        deger = _normalize(eslesme.group(1))
        if deger is None:
            continue
        onceki = metin[max(0, eslesme.start() - 3):eslesme.start()]
        bulunanlar.append((deger, bool(YUZDE_ONEKI.search(onceki))))
    return bulunanlar


def _duzlestir(veri, toplanan: set[float]) -> None:
    """İç içe araç çıktısındaki bütün sayısal değerleri toplar.

    METİN ALANLARI DA TARANIR. Bu bir incelik değil, yanlış alarmın ana kaynağıydı:
    kampanya adı "Faturaya 50 TL İade", açıklaması "%8 nakit iade" gibi sayıları
    metin içinde taşıyor. Sadece sayısal alanlara bakınca LLM bu rakamları
    doğru şekilde alıntıladığında denetçi "uydurma" diyordu.
    """
    if isinstance(veri, bool):
        return
    if isinstance(veri, str):
        for deger, _ in sayilari_cikar(veri):
            toplanan.add(deger)
    elif isinstance(veri, (int, float)):
        # NaN/inf araç çıktısında görülebiliyor (boş seriden alınan ortalama).
        # round(nan) ValueError atıyor; denetçi veri yüzünden çökmemeli.
        if math.isfinite(veri):
            toplanan.add(float(veri))
    elif isinstance(veri, dict):
        for deger in veri.values():
            _duzlestir(deger, toplanan)
    elif isinstance(veri, (list, tuple)):
        for deger in veri:
            _duzlestir(deger, toplanan)


def arac_degerleri(tool_calls: list[dict]) -> set[float]:
    """Araç çıktılarındaki tüm sayılar + LLM'in makul türevleri.

    Türevler ekleniyor çünkü LLM oranı yüzdeye çevirip yuvarlıyor: araç
    `pay: 0.2477` diyor, metinde "%25" görünüyor. Bu meşru bir dönüşüm, ihlal
    değil.
    """
    ham: set[float] = set()
    for cagri in tool_calls:
        _duzlestir(cagri.get("result"), ham)
        _duzlestir(cagri.get("arguments"), ham)

    genisletilmis: set[float] = set()
    for deger in ham:
        genisletilmis.add(deger)
        genisletilmis.add(round(deger))
        genisletilmis.add(round(deger, 1))
        genisletilmis.add(round(deger, 2))
        # Oran -> yüzde. MUTLAK değer alınıyor çünkü düşüşler metinde işaretsiz
        # yazılıyor: `degisim_orani: -0.78` cümlede "%78 azaldı" oluyor.
        # Üst sınır 1.0 değil ORAN_UST_SINIRI: %100'ü aşan artışlarda oran 1'i
        # geçiyor (`1.0733` -> "%107"). Sınırsız bırakmak denetimi zayıflatırdı,
        # çünkü her sayının 100 katı da havuza girerdi.
        if abs(deger) <= ORAN_UST_SINIRI:
            yuzdelik = abs(deger) * 100
            genisletilmis.add(yuzdelik)
            genisletilmis.add(round(yuzdelik))
            genisletilmis.add(round(yuzdelik, 1))
        genisletilmis |= _yuvarlama_varyantlari(deger)
    return genisletilmis


def _yuvarlama_varyantlari(deger: float) -> set[float]:
    """LLM'in yapabileceği yuvarlamalar: 53.609,49 -> 53.610 / 53.600 / 54.000.

    Toleransı gevşetmek yerine varyant üretmenin sebebi: tolerans TÜM sayı
    eksenini genişletir ve uydurma tutarları da kapsar; varyant yalnızca
    gerçekten var olan değerin makul yazımlarını ekler.
    """
    varyantlar: set[float] = set()
    for basamak in (10, 100, 1000):
        if abs(deger) >= basamak:
            varyantlar.add(round(deger / basamak) * basamak)
    return varyantlar


def _eslesiyor(deger: float, havuz: set[float]) -> bool:
    for aday in havuz:
        fark = abs(deger - aday)
        if fark <= MUTLAK_TOLERANS or fark <= abs(aday) * BAGIL_TOLERANS:
            return True
    return False


# --------------------------------------------------------------------------
# Üç denetim
# --------------------------------------------------------------------------


def izlenebilirlik_denetimi(metin: str, tool_calls: list[dict]) -> list[float]:
    """Araç çıktılarına dayanmayan sayıları döner (uydurma tespiti)."""
    havuz = arac_degerleri(tool_calls)
    izlenemeyen: list[float] = []
    for deger, _ in sayilari_cikar(metin):
        if deger in MUAF_SAYILAR or _eslesiyor(deger, havuz):
            continue
        izlenemeyen.append(deger)
    return izlenemeyen


def odul_denetimi(metin: str, tool_calls: list[dict]) -> list[str]:
    """Metindeki ödül oranı, önerilen kampanyanınkiyle uyuşuyor mu.

    ATIF hatasını yakalayan denetim budur. "%25 nakit iade" cümlesindeki 25
    araç çıktılarında var (müşterinin market payı), o yüzden izlenebilirlik
    denetiminden geçer — ama önerilen kampanyanın ödülü 8'dir. Burada ödül
    ifadesinin YANINDAKİ sayıya bakıp kampanyanın kendi `reward_value`'suyla
    karşılaştırıyoruz.
    """
    odul_kelimeleri = r"(?:nakit\s+iade|iade|indirim|puan|cashback)"
    desen = re.compile(rf"%\s*(\d+(?:[.,]\d+)?)\s*(?:oranında\s+)?{odul_kelimeleri}",
                       re.IGNORECASE)

    gecerli: set[float] = set()
    for cagri in tool_calls:
        sonuc = cagri.get("result") or {}
        if cagri.get("name") == "get_campaign_details" and "reward_value" in sonuc:
            gecerli.add(float(sonuc["reward_value"]))
    if not gecerli:
        return []                       # kıyaslanacak katalog verisi yok

    ihlaller = []
    for eslesme in desen.finditer(metin):
        deger = _normalize(eslesme.group(1))
        if deger is None:
            continue
        if not any(abs(deger - g) <= 0.01 for g in gecerli):
            ihlaller.append(
                f"odul_uyusmazligi: metinde %{eslesme.group(1)} yazıyor, "
                f"kampanya ödülleri {sorted(gecerli)}"
            )
    return ihlaller


def tavsiye_denetimi(metin: str) -> list[str]:
    """Finansal yönlendirme ifadesi var mı (düzenlemeye tabi faaliyet)."""
    kucuk = metin.lower()
    return [f"tavsiye_ifadesi: {kalip}" for kalip in TAVSIYE_KALIPLARI if kalip in kucuk]


def kampanya_denetimi(metin: str, tool_calls: list[dict], katalog: set[str]) -> list[str]:
    """Metinde geçen kampanya kodları araç çıktısındakilerle sınırlı mı."""
    izinli = {
        oneri["campaign_id"]
        for cagri in tool_calls if cagri.get("name") == "score_campaigns"
        for oneri in (cagri.get("result") or {}).get("oneriler", [])
    }
    izinli |= {
        (cagri.get("arguments") or {}).get("campaign_id")
        for cagri in tool_calls if cagri.get("name") == "get_campaign_details"
    }
    gecenler = {kod for kod in katalog if kod in metin}
    return [f"izinsiz_kampanya: {kod}" for kod in sorted(gecenler - izinli)]


def denetle(metin: str, tool_calls: list[dict], katalog: set[str] | None = None) -> DenetimSonucu:
    """Dört denetimi birden koşar."""
    sonuc = DenetimSonucu()

    izlenemeyen = izlenebilirlik_denetimi(metin, tool_calls)
    if izlenemeyen:
        sonuc.izlenemeyen_sayilar = izlenemeyen
        sonuc.ekle(f"izlenemeyen_sayi: {izlenemeyen}")

    for ihlal in odul_denetimi(metin, tool_calls):
        sonuc.ekle(ihlal)
    for ihlal in tavsiye_denetimi(metin):
        sonuc.ekle(ihlal)
    if katalog:
        for ihlal in kampanya_denetimi(metin, tool_calls, katalog):
            sonuc.ekle(ihlal)

    return sonuc

# --------------------------------------------------------------------------
# LLM'e gidecek veriyi maskeleme ve inceltme
# --------------------------------------------------------------------------

#: LLM payload'ından çıkarılacak anahtarlar. `validate.PII_COLUMNS`'tan türer —
#: tek bir PII listesi olsun, iki yerde ayrı ayrı bakım yapılmasın.
#:
#: `name` MUAF: kampanyanın adıdır, bir kişinin adı değil ("Market Alışverişine
#: Nakit İade"). Bunu da elemek agent'ın kampanyayı adıyla anmasını imkânsız
#: kılardı. `schemas.Customer` zaten hiçbir kişisel alan taşımıyor; buradaki
#: eleme, gerçek veriye geçişte `SELECT *` kazasına karşı ikinci ağdır.
MASKELENECEK_ANAHTARLAR: frozenset[str] = frozenset(PII_COLUMNS) - {"name"}

#: Payload inceltme varsayılanları. Amaç çift: (1) ücretsiz katmanda dakikada
#: 8.000 token sınırı var ve tek bir öneri akışı bunu dolduruyordu, (2) daha az
#: token = daha az gecikme. Modelin 5 cümlelik bir metin kurmak için 12 aylık
#: serinin tamamına ve 12 kategoriye ihtiyacı yok.
PAYLOAD_VARSAYILAN: dict[str, int] = {
    "max_categories": 6,
    "max_series_months": 6,
    "max_recurring": 3,
    "max_anomalies": 3,
}

#: Kırpılacak liste alanları -> hangi uçtan kırpılacağı ("bas" = baştan al,
#: "son" = sondan al). Aylık seride son aylar anlamlı, kategori listesinde ise
#: liste zaten paya göre sıralı olduğu için baş taraf.
KIRPILACAK_ALANLAR: dict[str, tuple[str, str]] = {
    "kategori_dagilimi": ("max_categories", "bas"),
    "aylik_seri": ("max_series_months", "son"),
    "duzenli_giderler": ("max_recurring", "bas"),
    "olagandisi_artislar": ("max_anomalies", "bas"),
}


def mask_payload(veri, ayar: dict | None = None):
    """LLM'e gitmeden önce payload'ı maskeler ve inceltir.

    İki iş bilinçli olarak tek fonksiyonda: ikisi de "modele ne gösteriyoruz"
    sorusunun cevabı ve tek bir yerden geçmeleri, denetlenebilir olmalarını
    sağlıyor. Çıktısı hem LLM'e gider hem de `cagri_kaydi`'na yazılır — yani
    denetçi de audit log da modelin GERÇEKTEN gördüğü veriyi görür. Denetimi
    tam payload'a karşı yapmak yanıltıcı olurdu: modele göstermediğimiz bir
    kategoriden rakam üretirse bu uydurmadır ve yakalanmalıdır.
    """
    limitler = {**PAYLOAD_VARSAYILAN, **(ayar or {})}

    def _gez(dugum, anahtar: str | None = None):
        if isinstance(dugum, dict):
            return {
                k: _gez(v, k)
                for k, v in dugum.items()
                if k.lower().strip() not in MASKELENECEK_ANAHTARLAR
            }
        if isinstance(dugum, list):
            kirpma = KIRPILACAK_ALANLAR.get(anahtar or "")
            if kirpma:
                limit_adi, uc = kirpma
                n = limitler[limit_adi]
                # n == 0 ayrı ele alınmak zorunda: Python'da `liste[-0:]` boş
                # liste değil TÜM listeyi döndürür, yani "bu alanı hiç gönderme"
                # ayarı tam tersine çalışıp payload'ı büyütüyordu.
                if n <= 0:
                    dugum = []
                else:
                    dugum = dugum[-n:] if uc == "son" else dugum[:n]
            return [_gez(e) for e in dugum]
        return dugum

    return _gez(veri)
