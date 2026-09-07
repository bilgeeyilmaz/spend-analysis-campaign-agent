"""Streamlit demo arayüzü — servisin bir istemcisi.

    streamlit run ui/streamlit_app.py          # API ayrı çalışıyor olmalı
    ./scripts/demo_ui.sh                       # ikisini birlikte başlatır

Arayüz iş kuralı içermez ve modele/ajana doğrudan erişmez; her şeyi HTTP
üzerinden ister. Bunun sebebi mimari iddianın kendisidir: ajan bir servistir,
arayüz onun tüketicilerinden biridir.

Yerleşim modern bir finans uygulamasının mantığını izler: tek büyük rakam (dönem
harcaması), onun altında ikonlu kategori satırları ve pay çubukları, sonra zaman
serisi. Grafik yığını yerine satır listesi tercih edildi — kategori dağılımı
insanın "neye ne kadar" diye okuduğu bir listedir, bar grafiği aynı bilgiyi daha
fazla mürekkeple verir.

Renk: kurumsal sarı/siyah/beyaz. Sarının tuzağı ölçülmüştür — `#FFD100` beyaz
zeminde 1,46:1 kontrast verir ve veri işareti olarak okunmaz; siyah zeminde
12,84:1'dir. Bu yüzden sarı **dolgu ve vurgu** rengidir (üstüne siyah yazı gelir),
açık temada veri işaretleri koyu altın (`#A67C00`, 3,82:1) ile çizilir, koyu
temada marka sarısının kendisiyle. Kural `tests/test_ui.py` içinde ölçülerek
sınanır — palet "güzel göründüğü için" değil, geçtiği için burada.
"""

from __future__ import annotations

import html

import httpx
import plotly.graph_objects as go
import streamlit as st

VARSAYILAN_API = "http://127.0.0.1:8000"

PALET = {
    "light": {
        "yuzey": "#FFFFFF",
        "kart": "#FAFAF7",
        "ink": "#12120F",
        "ikincil": "#3D3D38",
        "izgara": "#EDECE6",
        "seri": "#A67C00",       # veri işareti — beyaz zeminde 3.82:1
        "vurgu": "#12120F",      # seçili işaret
        "marka": "#FFD100",      # dolgu/şerit; üstüne siyah yazı
        "marka_ink": "#12120F",
        "artis": "#B3261E",      # harcama artışı: iyi haber değil
        "azalis": "#1B5E20",
    },
    "dark": {
        "yuzey": "#12120F",
        "kart": "#1D1D19",
        "ink": "#FFFFFF",
        "ikincil": "#C3C2B7",
        "izgara": "#33322F",
        "seri": "#FFD100",       # koyu zeminde marka sarısı 12.84:1
        "vurgu": "#FFFFFF",
        "marka": "#FFD100",
        "marka_ink": "#12120F",
        "artis": "#F2B8B5",
        "azalis": "#A5D6A7",
    },
}

#: Kategori -> renk. **Sabit eşleme**: renk kimliği takip eder, sıralamayı değil.
#: Önce en büyük dilime ilk ton veriliyordu; o zaman aynı kategori müşteri
#: değiştikçe renk değiştiriyordu ve iki ekranı yan yana koyunca hiçbir şey
#: karşılaştırılamıyordu. Market her müşteride yeşil.
#: Koyu tema aynı renklerin koyu zemine göre basamakları — ters çevrilmiş kopya
#: değil. Her iki sütun da kontrast eşiğine karşı ölçülüyor (`tests/test_ui.py`).
KATEGORI_RENGI = {
    "light": {
        "market": "#1F6F5C", "akaryakit": "#B58900", "restoran": "#B0483A",
        "giyim": "#8E4585", "elektronik": "#3E5060", "seyahat": "#2E7D9A",
        "saglik": "#2E7D32", "egitim": "#5B4FA8", "eglence": "#C05621",
        "telekom": "#5A7183", "online_alisveris": "#7A5CA8", "diger": "#6B6B63",
    },
    "dark": {
        "market": "#4FBFA0", "akaryakit": "#FFD100", "restoran": "#F08C7D",
        "giyim": "#D68FCB", "elektronik": "#9DB3C4", "seyahat": "#6FC5E0",
        "saglik": "#7BC97F", "egitim": "#A79BEA", "eglence": "#F0A05A",
        "telekom": "#B8C4CE", "online_alisveris": "#BFA5E8", "diger": "#A8A89E",
    },
}

#: Toplanan artık dilim ("Diğer N kategori") hiçbir kategoriye ait değil; nötr.
ARTIK_RENGI = {"light": "#7A7A72", "dark": "#8E8E85"}

#: İşaret rengi için asgari kontrast (WCAG grafik nesnesi), metin için 4.5:1.
ISARET_ESIGI = 3.0
METIN_ESIGI = 4.5

#: Kategori kimliği -> (simge, okunur ad). Kimlikler İngilizce ve şemadan gelir;
#: kullanıcı bunları görmemeli. `tests/test_ui.py` sözlüğün eksiksizliğini sınar.
KATEGORI_GORUNUM: dict[str, tuple[str, str]] = {
    "market": ("🛒", "Market"),
    "akaryakit": ("⛽", "Akaryakıt"),
    "restoran": ("🍽️", "Restoran"),
    "giyim": ("👕", "Giyim"),
    "elektronik": ("💻", "Elektronik"),
    "seyahat": ("✈️", "Seyahat"),
    "saglik": ("🏥", "Sağlık"),
    "egitim": ("🎓", "Eğitim"),
    "eglence": ("🎬", "Eğlence"),
    "telekom": ("📱", "Telekom"),
    "online_alisveris": ("📦", "Online alışveriş"),
    "diger": ("💳", "Diğer harcamalar"),
}


def gorunum(kategori: str) -> tuple[str, str]:
    return KATEGORI_GORUNUM.get(kategori, ("•", kategori.replace("_", " ").capitalize()))


# --------------------------------------------------------------------------
# Renk ve biçim
# --------------------------------------------------------------------------


def _bagil_parlaklik(hex_renk: str) -> float:
    h = hex_renk.lstrip("#")
    kanallar = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    kanallar = [k / 12.92 if k <= 0.03928 else ((k + 0.055) / 1.055) ** 2.4 for k in kanallar]
    return 0.2126 * kanallar[0] + 0.7152 * kanallar[1] + 0.0722 * kanallar[2]


def kontrast_orani(on: str, arka: str) -> float:
    """WCAG kontrast oranı. Renk kararlarını gözle değil bununla veriyoruz."""
    a, b = _bagil_parlaklik(on), _bagil_parlaklik(arka)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def rgba(hex_renk: str, saydamlik: float) -> str:
    """'#A67C00' -> 'rgba(166,124,0,0.1)'. Dolgu rengi seri renginden türesin."""
    h = hex_renk.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{saydamlik})"


def tema_modu() -> str:
    """Aktif tema adı; okunamazsa açık tema."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tema() -> dict[str, str]:
    """Streamlit'in aktif teması; okunamazsa açık tema varsayılır."""
    return PALET[tema_modu()]


def tl(deger: float) -> str:
    """1234567.8 -> '1.234.568 TL' (Türkçe ayraçlarla)."""
    return f"{deger:,.0f}".replace(",", ".") + " TL"


# --------------------------------------------------------------------------
# Servis istemcisi
# --------------------------------------------------------------------------


def _kok() -> str:
    return st.session_state.get("api_kok", VARSAYILAN_API).rstrip("/")


def istek(yol: str, govde: dict | None = None, zaman_asimi: float = 180.0) -> dict:
    """Tek çıkış noktası: hata mesajlarını arayüzde okunur hâle getirir."""
    url = f"{_kok()}{yol}"
    try:
        cevap = (httpx.get(url, timeout=zaman_asimi) if govde is None
                 else httpx.post(url, json=govde, timeout=zaman_asimi))
    except httpx.RequestError as hata:
        raise RuntimeError(
            f"Servise ulaşılamadı ({url}).\n\nBaşlatmak için:  `uvicorn src.api.main:app`"
        ) from hata

    if cevap.status_code >= 400:
        try:
            ayrinti = cevap.json().get("detail", cevap.text)
        except Exception:
            ayrinti = cevap.text
        raise RuntimeError(f"HTTP {cevap.status_code}: {ayrinti}")
    return cevap.json()


@st.cache_data(ttl=60, show_spinner=False)
def saglik(kok: str) -> dict:
    return httpx.get(f"{kok}/health", timeout=10).json()


@st.cache_data(ttl=300, show_spinner=False)
def musteri_listesi(kok: str, yalniz_izinli: bool) -> list[dict]:
    yol = "/customers?limit=2000" + ("&opted_in=true" if yalniz_izinli else "")
    return httpx.get(f"{kok}{yol}", timeout=30).json()["musteriler"]


@st.cache_data(ttl=300, show_spinner=False)
def harcama(kok: str, cid: str, aylar: int) -> dict:
    return httpx.get(f"{kok}/customers/{cid}/spending?months={aylar}", timeout=60).json()


@st.cache_data(ttl=300, show_spinner=False)
def katalog(kok: str) -> list[dict]:
    return httpx.get(f"{kok}/campaigns?active_only=false", timeout=30).json()["kampanyalar"]


@st.cache_data(ttl=300, show_spinner=False)
def profil(kok: str, cid: str) -> dict:
    return httpx.get(f"{kok}/customers/{cid}/profile", timeout=30).json()


# --------------------------------------------------------------------------
# Görsel bileşenler
# --------------------------------------------------------------------------


def stil(renk: dict) -> str:
    return f"""
    <style>
      .blok {{
        background: {renk['kart']};
        border: 1px solid {renk['izgara']};
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 14px;
      }}
      .hero-etiket {{ color: {renk['ikincil']}; font-size: .82rem;
                     letter-spacing: .04em; text-transform: uppercase; }}
      .hero-rakam  {{ color: {renk['ink']}; font-size: 2.6rem; font-weight: 700;
                     line-height: 1.15; margin: 2px 0 6px; }}
      .hero-alt    {{ color: {renk['ikincil']}; font-size: .9rem; }}
      .rozet {{ display:inline-block; padding: 3px 10px; border-radius: 999px;
                font-size: .82rem; font-weight: 600; }}
      .satir {{ display:flex; align-items:center; gap:12px; padding:10px 0;
                border-bottom:1px solid {renk['izgara']}; }}
      .satir:last-child {{ border-bottom:none; }}
      .satir-simge {{ font-size:1.25rem; width:34px; height:34px; border-radius:10px;
                      background:{rgba(renk['seri'], 0.14)};
                      display:flex; align-items:center; justify-content:center; }}
      .satir-orta {{ flex:1; min-width:0; }}
      .satir-ad   {{ color:{renk['ink']}; font-weight:600; font-size:.95rem; }}
      .satir-cubuk {{ height:6px; border-radius:999px;
                      background:{renk['izgara']}; margin-top:6px; }}
      .satir-dolgu {{ height:6px; border-radius:999px; background:{renk['seri']}; }}
      .satir-sag  {{ text-align:right; white-space:nowrap; }}
      .satir-tutar {{ color:{renk['ink']}; font-weight:600; font-size:.95rem; }}
      .satir-pay  {{ color:{renk['ikincil']}; font-size:.8rem; }}
      .marka-serit {{ background:{renk['marka']}; color:{renk['marka_ink']};
                      padding:16px 22px; border-radius:16px; margin-bottom:14px; }}
      .marka-serit h1 {{ font-size:1.55rem; margin:0; font-weight:700; }}
      .marka-serit p  {{ margin:3px 0 0; font-size:.9rem; opacity:.8; }}
      button[data-testid="stBaseButton-primary"] {{
        background:{renk['marka']}; color:{renk['marka_ink']};
        border:1px solid {renk['marka']}; font-weight:600; border-radius:999px;
      }}
    </style>
    """


def trend_rozeti(degisim: float | None, renk: dict) -> str:
    """Harcama artışı kırmızı, azalış yeşil — 'iyi/kötü' değil, alışılmış yön."""
    if degisim is None:
        return ""
    artis = degisim > 0
    ton = renk["artis"] if artis else renk["azalis"]
    ok = "▲" if artis else "▼"
    return (f'<span class="rozet" style="background:{rgba(ton, 0.14)};color:{ton}">'
            f'{ok} %{abs(degisim) * 100:.0f}</span>')


def hero_karti(a: dict, aylar: int, renk: dict) -> str:
    """Dönem özeti.

    İki ayrı zaman ölçeği var ve ikisi de **açıkça etiketlenmeli**: büyük rakam
    seçili dönemin (3/6/12 ay) toplamı, alt satırdaki rozet ise son 30 günün
    önceki iki ayın ortalamasına göre değişimi. Etiketsiz tek satıra
    sıkıştırıldığında "12 ayda %2 düşmüş" diye okunuyordu — oysa %2, son ayın
    kıyası. Zaman ölçeği karışan bir özet, özet olmaktan çıkar.
    """
    trend = a.get("trend") or {}
    degisim = trend.get("degisim_orani")
    rozet = trend_rozeti(degisim, renk)
    kiyas = (f'{rozet} <span style="color:{renk["ikincil"]}">önceki iki ayın '
             "aylık ortalamasına göre</span>") if rozet else ""
    return f"""
    <div class="blok">
      <div class="hero-etiket">Son {aylar} ayda harcama</div>
      <div class="hero-rakam">{tl(a['toplam_harcama'])}</div>
      <div class="hero-alt">
        {aylar} ayda {a['islem_adedi']} işlem · ortalama sepet {tl(a['ortalama_sepet'])}
      </div>
      <div class="hero-alt" style="margin-top:8px">
        Son 30 gün: <b>{tl(trend.get('son_30_gun', 0))}</b> &nbsp; {kiyas}
      </div>
    </div>
    """


def kategori_satirlari(dagilim: list[dict],
                       karsilastirma: list[dict] | None = None) -> list[dict]:
    """Kategori dağılımını çizime hazır satırlara çevirir.

    Çubuk genişliği **yazan payın kendisidir**. Önce en büyük kategoriye göre
    ölçekleniyordu; o zaman satırda "%24" yazarken çubuk tam dolu görünüyordu ve
    ikisi birbiriyle çelişiyordu. Çubuk bir karşılaştırma aracı değil, yazan
    sayının görsel karşılığı olmalı.

    Çok küçük paylar için asgari bir genişlik bırakılıyor; yoksa %1'lik kategori
    hiç çubuğu yokmuş gibi görünür.
    """
    veri = sorted(dagilim, key=lambda d: d["tutar"], reverse=True)
    if not veri:
        return []
    degisim = {k["kategori"]: k["degisim_orani"] for k in (karsilastirma or [])}
    satirlar = []
    for d in veri:
        simge, ad = gorunum(d["kategori"])
        satirlar.append({
            "kategori": d["kategori"],
            "simge": simge,
            "ad": ad,
            "tutar": d["tutar"],
            "pay": d["pay"],
            "islem_adedi": d["islem_adedi"],
            "genislik": round(max(d["pay"] * 100, 1.5), 1),
            # None = önceki dönemde bu kategoride harcama yok ("yeni").
            "degisim": degisim.get(d["kategori"]),
            "yeni": d["kategori"] in degisim and degisim[d["kategori"]] is None,
        })
    return satirlar


def degisim_rozeti(degisim: float | None, yeni: bool, renk: dict) -> str:
    """Kategori satırındaki dönem karşılaştırma rozeti.

    %5'in altındaki fark rozetlenmiyor: her satıra bir ok koymak listeyi
    gürültüye boğar ve gerçek hareketi görünmez yapar.
    """
    if yeni:
        return (f'<span class="rozet" style="background:{rgba(renk["seri"], 0.16)};'
                f'color:{renk["ikincil"]}">yeni</span>')
    if degisim is None or abs(degisim) < 0.05:
        return ""
    ton = renk["artis"] if degisim > 0 else renk["azalis"]
    ok = "▲" if degisim > 0 else "▼"
    return (f'<span class="rozet" style="background:{rgba(ton, 0.14)};color:{ton}">'
            f'{ok} %{abs(degisim) * 100:.0f}</span>')


def kategori_listesi(dagilim: list[dict], renk: dict, secili: str | None = None,
                     karsilastirma: list[dict] | None = None) -> str:
    parcalar = ['<div class="blok">']
    for s in kategori_satirlari(dagilim, karsilastirma):
        vurgu = s["kategori"] == secili
        dolgu = renk["vurgu"] if vurgu else renk["seri"]
        parcalar.append(f"""
          <div class="satir">
            <div class="satir-simge">{s['simge']}</div>
            <div class="satir-orta">
              <div class="satir-ad">{html.escape(s['ad'])}
                {degisim_rozeti(s.get('degisim'), s.get('yeni', False), renk)}</div>
              <div class="satir-cubuk">
                <div class="satir-dolgu" style="width:{s['genislik']}%;background:{dolgu}"></div>
              </div>
            </div>
            <div class="satir-sag">
              <div class="satir-tutar">{tl(s['tutar'])}</div>
              <div class="satir-pay">%{s['pay'] * 100:.0f} · {s['islem_adedi']} işlem</div>
            </div>
          </div>
        """)
    parcalar.append("</div>")
    return "".join(parcalar)


def isyeri_adi(merchant_id: str, kategori: str) -> str:
    """Ham kimliği okunur bir ada çevirir: `M_ELEKTRONIK_018` -> "Elektronik #018".

    Sentetik veride üye işyeri **adı** yok, yalnız kimlik var; ekrana ham kimlik
    basmak listeyi anlamsız gösteriyordu. Gerçek veriye geçildiğinde bu fonksiyon
    yerini kaynaktaki gerçek işyeri adına bırakır — o yüzden biçim burada, tek
    yerde duruyor.
    """
    _, kategori_adi = gorunum(kategori)
    numara = merchant_id.rsplit("_", 1)[-1]
    return f"{kategori_adi} #{numara}" if numara.isdigit() else merchant_id


def uye_isyeri_listesi(isyerleri: list[dict], renk: dict, kategori: str | None = None,
                       tonlar: dict[str, str] | None = None) -> str:
    """Kategori "neye", üye işyeri "nereye" harcadığını söyler.

    Çubuk rengi işyerinin kategorisinin halka grafikteki tonudur: iki görsel
    aynı renk diliyle konuşsun, kullanıcı "bu mağaza en büyük dilimimde" diye
    okuyabilsin.
    """
    veri = [m for m in isyerleri if not kategori or m["kategori"] == kategori]
    if not veri:
        return f'<div class="blok"><div class="hero-alt">Bu seçimde işlem yok.</div></div>'

    en_buyuk = max(m["tutar"] for m in veri) or 1
    tonlar = tonlar or {}
    parcalar = ['<div class="blok">']
    for m in veri:
        simge, _ = gorunum(m["kategori"])
        ton = tonlar.get(m["kategori"], renk["seri"])
        parcalar.append(f"""
          <div class="satir">
            <div class="satir-simge">{simge}</div>
            <div class="satir-orta">
              <div class="satir-ad">{html.escape(isyeri_adi(m['merchant_id'], m['kategori']))}</div>
              <div class="satir-cubuk">
                <div class="satir-dolgu" style="width:{m['tutar'] / en_buyuk * 100:.1f}%;
                     background:{ton}"></div>
              </div>
            </div>
            <div class="satir-sag">
              <div class="satir-tutar">{tl(m['tutar'])}</div>
              <div class="satir-pay">{m['islem_adedi']} işlem</div>
            </div>
          </div>
        """)
    parcalar.append("</div>")
    return "".join(parcalar)


def halka_dilimleri(dagilim: list[dict], n: int = 5) -> list[dict]:
    """En büyük n kategori + kalanı "Diğer".

    12 dilimlik bir halka okunmaz ve 12 ayrı renk paletin ayırt edilebilirlik
    sınırını aşar. Kalanı tek dilimde toplamak bilgi kaybı değil: tam liste
    hemen yanındaki satırlarda zaten duruyor.
    """
    veri = sorted(dagilim, key=lambda d: d["tutar"], reverse=True)
    dilimler = [{"ad": gorunum(d["kategori"])[1], "tutar": d["tutar"],
                 "kategori": d["kategori"]} for d in veri[:n]]
    kalanlar = veri[n:]
    kalan_tutar = sum(d["tutar"] for d in kalanlar)
    if kalan_tutar > 0:
        # Toplanan dilime "Diğer" DEMİYORUZ: veri setinde zaten `diger` adında
        # gerçek bir kategori var (bilinen gruplara girmeyen harcamalar) ve o
        # kategori ilk beşe girdiğinde efsanede iki ayrı "Diğer" görünüyordu.
        # `kategori: None` = bu dilim bir kategori değil, toplanmış artık.
        # Rengi etiket metninden çıkarmak yanlıştı: gerçek `diger` kategorisinin
        # adı da "Diğer harcamalar" olduğu için ikisi aynı griye boyanıyordu.
        dilimler.append({
            "ad": f"Diğer {len(kalanlar)} kategori",
            "tutar": round(kalan_tutar, 2),
            "kategori": None,
        })
    return dilimler


def kategori_tonlari(dagilim: list[dict], mod: str) -> dict[str, str]:
    """Kategori -> renk. Sıralamadan bağımsız, sabit eşleme."""
    return {d["kategori"]: KATEGORI_RENGI[mod].get(d["kategori"], ARTIK_RENGI[mod])
            for d in dagilim}


def halka_grafigi(dagilim: list[dict], toplam: float, renk: dict, mod: str) -> go.Figure:
    dilimler = halka_dilimleri(dagilim)
    # Dilim rengi kategorinin sabit rengi; artık dilim (kategori=None) nötr.
    renkler = [
        KATEGORI_RENGI[mod].get(d["kategori"], ARTIK_RENGI[mod]) if d["kategori"]
        else ARTIK_RENGI[mod]
        for d in dilimler
    ]

    fig = go.Figure(go.Pie(
        labels=[d["ad"] for d in dilimler],
        values=[d["tutar"] for d in dilimler],
        hole=0.62,
        sort=False,
        direction="clockwise",
        # Ayırıcı çizgi 1px: 2px'te açık gri dilimler "çerçeveli boşluk" gibi
        # görünüyordu. Renkler zaten birbirinden ayrı olduğu için kalın ayırıcıya
        # gerek yok.
        marker=dict(colors=renkler, line=dict(color=renk["yuzey"], width=1)),
        textinfo="none",                       # dilim üstü yazı halkayı boğar
        hovertemplate="%{label}: %{value:,.0f} TL (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{tl(toplam)}</b>", showarrow=False,
        font=dict(size=20, color=renk["ink"]),
    )
    fig.update_layout(
        height=260, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=renk["ikincil"], size=12),
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(color=renk["ikincil"])),
    )
    return fig


def odul_metni(deger: float | None, birim: str | None) -> str:
    """Ödülü kendi birimiyle yazar.

    Katalogda üç birim var: `yuzde`, `tutar`, `taksit_sayisi`. Yüzde olmayan her
    şeyi TL saymak "6 taksit" kampanyasını **"6 TL"** diye gösteriyordu — hem
    yanlış hem de kampanyayı değersiz gösteren bir hata.
    """
    if deger is None or birim is None:
        return "—"
    if birim == "yuzde":
        return f"%{deger:.0f}"
    if birim == "taksit_sayisi":
        return f"{deger:.0f} taksit"
    return tl(deger)


def onerilen_kampanyalar(cevap: dict, katalog: list[dict], n: int = 3) -> list[dict]:
    """Ajanın araç izinden skorları, katalogdan detayları alıp birleştirir.

    Skorlama aracı yalnız kimlik/ad/skor döndürüyor; ödül oranı ve asgari harcama
    katalogda. Sunum alanlarını skorlama aracına eklemek yerine burada
    birleştiriyoruz: o araç LLM'e gidiyor ve her ek alan metne sızma yüzeyi.
    """
    skorlar = []
    for cagri in cevap.get("tool_calls", []):
        if cagri["name"] == "score_campaigns":
            skorlar = (cagri.get("result") or {}).get("oneriler", []) or []
    if not skorlar:
        return []

    detay = {k["campaign_id"]: k for k in katalog}
    kartlar = []
    for o in skorlar[:n]:
        k = detay.get(o["campaign_id"], {})
        kartlar.append({
            "campaign_id": o["campaign_id"],
            "ad": k.get("name") or o.get("name", o["campaign_id"]),
            "aciklama": k.get("description", ""),
            "odul": odul_metni(k.get("reward_value"), k.get("reward_unit")),
            "odul_tipi": k.get("reward_type", ""),
            "min_spend": k.get("min_spend"),
            "son_tarih": str(k.get("valid_to", ""))[:10],
            "kategoriler": k.get("target_categories", []),
            "skor": o.get("score"),
        })
    return kartlar


def kampanya_kartlari(kartlar: list[dict], renk: dict) -> str:
    parcalar = []
    for k in kartlar:
        simge = gorunum(k["kategoriler"][0])[0] if k["kategoriler"] else "🎁"
        alt = " · ".join(x for x in [
            f"asgari {tl(k['min_spend'])}" if k.get("min_spend") else "",
            f"son {k['son_tarih']}" if k.get("son_tarih") else "",
        ] if x)
        parcalar.append(f"""
          <div class="blok" style="padding:14px 16px">
            <div class="satir" style="border:none;padding:0">
              <div class="satir-simge">{simge}</div>
              <div class="satir-orta">
                <div class="satir-ad">{html.escape(k['ad'])}</div>
                <div class="satir-pay">{html.escape(alt)}</div>
              </div>
              <div class="satir-sag">
                <span class="rozet" style="background:{renk['marka']};
                      color:{renk['marka_ink']}">{k['odul']}</span>
              </div>
            </div>
            <div class="hero-alt" style="margin-top:8px">
              {html.escape(k['aciklama'])}
            </div>
          </div>
        """)
    return "".join(parcalar)


def butce_durumu(harcanan: float, limit: float | None) -> dict | None:
    """Kullanıcının koyduğu dönem limitine göre doluluk.

    Limit **kullanıcının kendi kararıdır**; sistem limit önermez. Öneri vermek
    finansal tavsiye olurdu ve ajanın da arayüzün de yapmadığı şey budur.
    """
    if not limit or limit <= 0:
        return None
    oran = harcanan / limit
    return {
        "oran": round(oran, 4),
        "genislik": round(min(oran, 1.0) * 100, 1),
        "asildi": oran > 1.0,
        "kalan": round(limit - harcanan, 2),
    }


def aylik_grafik(seri: list[dict], renk: dict) -> go.Figure:
    """Zaman serisi: ince çizgi, yumuşak dolgu, eksen süsü yok."""
    fig = go.Figure(go.Scatter(
        x=[d["ay"] for d in seri],
        y=[d["tutar"] for d in seri],
        mode="lines+markers",
        line=dict(color=renk["seri"], width=2, shape="spline", smoothing=0.6),
        marker=dict(color=renk["seri"], size=8),
        fill="tozeroy",
        fillcolor=rgba(renk["seri"], 0.10),
        hovertemplate="%{x}: %{y:,.0f} TL<extra></extra>",
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=renk["ikincil"], size=13),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=renk["izgara"], zeroline=False, tickformat=",.0f")
    return fig


def denetim_durumu(cevap: dict) -> tuple[str, str]:
    """Cevabın denetim durumu: ("ok" | "duzeltildi" | "yedek", mesaj).

    `uyarilar` dolu olması metnin güvensiz olduğu anlamına GELMEZ — orkestratör
    ihlali kaydedip bir düzeltme denemesi yapıyor, düzeltme tutarsa metin
    denetimden geçmiş hâlde dönüyor ve uyarı listesi yakalananların kaydı olarak
    kalıyor. Ayrımı `fallback` verir.
    """
    if cevap["fallback"]:
        return "yedek", (
            "Cevap deterministik şablondan üretildi (LLM metni denetimden geçemedi "
            "ya da sağlayıcı yanıt veremedi): " + ", ".join(cevap["uyarilar"])
        )
    if cevap["uyarilar"]:
        return "duzeltildi", (
            "Denetim ilk taslakta sorun yakaladı ve metin düzelttirildi; "
            "gösterilen cevap denetimden geçti. Yakalananlar: "
            + ", ".join(cevap["uyarilar"])
        )
    return "ok", "Denetimden geçti: metindeki her sayı bir araç çıktısına dayanıyor."


def one_cikanlar(a: dict) -> list[str]:
    """Analizin cümleye dönüşmüş hâli — ham tablo okunmaz, cümle okunur."""
    satirlar = []
    for g in a.get("duzenli_giderler", [])[:3]:
        _, ad = gorunum(g["kategori"])
        satirlar.append(
            f"**{ad}** kategorisinde {g['ay_sayisi']} aydır aynı üye işyerine ayda "
            f"ortalama **{tl(g['ortalama_tutar'])}** ödüyorsunuz "
            f"(genelde ayın {g['tipik_gun']}'i civarı)."
        )
    for o in a.get("olagandisi_artislar", [])[:2]:
        _, ad = gorunum(o.get("kategori", "diger"))
        tutar = o.get("tutar") or o.get("fark") or 0
        satirlar.append(f"**{ad}** kategorisinde olağandışı bir hareket var: {tl(tutar)}.")
    trend = a.get("trend") or {}
    degisim = trend.get("degisim_orani")
    if degisim is not None and abs(degisim) >= 0.15:
        yon = "artmış" if degisim > 0 else "azalmış"
        satirlar.append(
            f"Son 30 gündeki harcamanız önceki aylık ortalamaya göre "
            f"**%{abs(degisim) * 100:.0f} {yon}**."
        )
    return satirlar


# --------------------------------------------------------------------------
# Ajan sekmesi
# --------------------------------------------------------------------------


ONERI_SORUSU = "Bana uygun bir kampanya var mı?"


def onerilen_sorular(a: dict) -> list[str]:
    """Müşterinin verisine göre hazır sorular.

    Sorular betimleyicidir ("ne kadar", "neler"), "neden" değil — ajan tarif eder,
    yorum ve tavsiye vermez; "neden arttı" sorusu onu sebep uydurmaya davet ederdi.
    """
    sorular = ["Son 3 ayda nereye harcadım?"]
    if a.get("duzenli_giderler"):
        sorular.append("Düzenli ödemelerim neler?")
    if a.get("olagandisi_artislar"):
        sorular.append("Olağandışı bir harcamam var mı?")
    ilk_uc = a.get("ilk_uc_kategori") or []
    if ilk_uc and len(sorular) < 4:
        sorular.append(f"{gorunum(ilk_uc[0])[1]} harcamam ne kadar?")
    return sorular[:4]


def hazir_soru_secildi(anahtar: str, durum) -> None:
    """Pill seçimini **tüketir**: soruyu kuyruğa alır ve widget'ı sıfırlar.

    `st.pills` seçimi kalıcıdır. Sıfırlanmazsa soru sorulup sayfa yeniden
    çalıştığında pill hâlâ seçili döner, kod soruyu bir kez daha tetikler ve
    arayüz sonsuz döngüye girip servise durmadan istek atar. (Ölçüldü: pill'e
    tıklandığında sayfa 300 sn'de tamamlanmadı; `st.chat_input` değerini yalnız
    bir kez döndürdüğü için yazarak sorma yolu aynı hatadan etkilenmiyordu.)

    Sıfırlama **`on_change` geri çağrısında** yapılmak zorunda: bir widget'ın
    kendi anahtarını, widget oluşturulduktan sonra betiğin gövdesinde değiştirmek
    `StreamlitAPIException` veriyor. Geri çağrı yeniden çalıştırmadan önce
    koştuğu için orada serbest. Aynı soruyu ikinci kez sorabilmek de buna bağlı.
    """
    secilen = durum.get(anahtar)
    if not secilen:
        return
    durum["bekleyen_soru"] = secilen
    durum[anahtar] = None


def _cevapla(cid: str, soru: str) -> None:
    """Soruyu servise sorar ve geçmişe yazar. Öneri de bir sorudur: aynı akış."""
    gecmis = st.session_state.setdefault("sohbet", {}).setdefault(cid, [])
    gecmis.append({"rol": "user", "metin": soru})
    try:
        cevap = (istek("/recommend", {"customer_id": cid}) if soru == ONERI_SORUSU
                 else istek("/chat", {"customer_id": cid, "question": soru}))
    except RuntimeError as hata:
        gecmis.append({"rol": "assistant", "metin": f"⚠️ {hata}"})
        return
    gecmis.append({"rol": "assistant", "metin": cevap["answer"], "cevap": cevap})


#: Müşteriye gösterilen kısa durum. Ayrıntı (hangi ihlal, hangi sayı) geliştirici
#: bilgisidir ve "Gerekçe" kutusunun içinde kalır: `izlenemeyen_sayi: [2263.0]`
#: müşterinin okuyacağı bir cümle değil. Rozetin kendisi kalıyor, çünkü cevabın
#: doğrulanmış olduğunu göstermek ürünün asıl iddiası.
DURUM_ROZETI = {
    "ok": ("✅", "Doğrulandı"),
    "duzeltildi": ("🛡️", "Doğrulandı (denetim düzeltti)"),
    "yedek": ("🛟", "Özet bilgi"),
}


def cevap_detay(cevap: dict) -> None:
    """Cevabın altındaki kısa durum + istenirse tam gerekçe."""
    seviye, mesaj = denetim_durumu(cevap)
    simge, etiket = DURUM_ROZETI[seviye]
    araclar = " → ".join(c["name"] for c in cevap["tool_calls"]) or "—"

    st.caption(f"{simge} {etiket}")
    with st.expander("Gerekçe — nasıl hesaplandı"):
        st.caption(f"Denetim: {mesaj}")
        st.caption(f"Araçlar: {araclar} · sağlayıcı: {cevap['provider']} · "
                   f"tur: {cevap['iterations']}")
        st.json(cevap["tool_calls"])


def sohbet_goster(cid: str, analiz: dict) -> None:
    if st.session_state.get("bekleyen_soru"):
        soru = st.session_state.pop("bekleyen_soru")
        with st.spinner("Agent araçları çağırıyor..."):
            _cevapla(cid, soru)

    gecmis = st.session_state.setdefault("sohbet", {}).setdefault(cid, [])

    ust_sol, ust_sag = st.columns([3, 1])
    with ust_sol:
        anahtar = f"hazir_{cid}"
        st.pills(
            "Hazır sorular", onerilen_sorular(analiz),
            selection_mode="single", key=anahtar, label_visibility="collapsed",
            on_change=hazir_soru_secildi, args=(anahtar, st.session_state),
        )
    with ust_sag:
        if st.button(f"🎯 {ONERI_SORUSU}", type="primary", width="stretch"):
            st.session_state["bekleyen_soru"] = ONERI_SORUSU
            st.rerun()

    for mesaj in gecmis:
        with st.chat_message(mesaj["rol"]):
            st.markdown(mesaj["metin"])
            if mesaj.get("cevap"):
                kartlar = onerilen_kampanyalar(mesaj["cevap"], katalog(_kok()))
                if kartlar:
                    st.html(kampanya_kartlari(kartlar, tema()))
                cevap_detay(mesaj["cevap"])

    if gecmis and st.button("Sohbeti temizle"):
        st.session_state["sohbet"][cid] = []
        st.rerun()

    soru = st.chat_input("Sorunuzu yazın…")
    if soru:
        st.session_state["bekleyen_soru"] = soru
        st.rerun()


# --------------------------------------------------------------------------
# Sayfa
# --------------------------------------------------------------------------


def analiz_goster(a: dict, aylar: int, renk: dict) -> None:
    if not a.get("islem_var"):
        st.info("Bu müşterinin seçilen dönemde işlemi yok.")
        return

    st.html(hero_karti(a, aylar, renk))

    # Kategori filtresi. Seçim yalnız görünümü daraltır — veri yeniden
    # hesaplanmaz, çünkü paylar bütünün içindeki payı gösterir.
    kategoriler = [d["kategori"] for d in a["kategori_dagilimi"][:8]]
    secim = st.pills(
        "Kategori filtresi",
        ["Tümü"] + [gorunum(k)[1] for k in kategoriler],
        selection_mode="single", default="Tümü", label_visibility="collapsed",
        key=f"filtre_{a['customer_id']}",
    )
    ad_to_kategori = {gorunum(k)[1]: k for k in kategoriler}
    secili = ad_to_kategori.get(secim or "Tümü")

    sol, sag = st.columns([1, 1])
    with sol:
        st.plotly_chart(
            halka_grafigi(a["kategori_dagilimi"], a["toplam_harcama"], renk, tema_modu()),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(f"Nereye harcadınız · önceki {aylar} aya göre")
        st.html(kategori_listesi(a["kategori_dagilimi"], renk, secili,
                                 a.get("kategori_karsilastirma")))
        if secili:
            butce_paneli(a, secili, aylar, renk)
    with sag:
        st.caption("Aylık toplam — son 12 ay")
        st.plotly_chart(aylik_grafik(a["aylik_seri"], renk),
                        width="stretch", config={"displayModeBar": False})
        st.caption("Nerelere harcadınız — en çok harcanan üye işyerleri"
                   + (f" · {gorunum(secili)[1]}" if secili else ""))
        st.html(uye_isyeri_listesi(
            a.get("ilk_uye_isyerleri", []), renk, secili,
            kategori_tonlari(a["kategori_dagilimi"], tema_modu()),
        ))

    satirlar = one_cikanlar(a)
    if satirlar:
        st.caption("Öne çıkanlar")
        for satir in satirlar:
            st.markdown(f"- {satir}")

    with st.expander("Detay tabloları"):
        st.dataframe(a["kategori_dagilimi"], width="stretch", hide_index=True)
        if a.get("kategori_karsilastirma"):
            st.dataframe(a["kategori_karsilastirma"], width="stretch", hide_index=True)
        st.dataframe(a["aylik_seri"], width="stretch", hide_index=True)


def butce_paneli(a: dict, kategori: str, aylar: int, renk: dict) -> None:
    """Kullanıcının kendi koyduğu dönem limiti ve doluluğu.

    Limit önerilmez, sorulur: bir limit **önermek** finansal tavsiye olurdu.
    """
    kayit = next((d for d in a["kategori_dagilimi"] if d["kategori"] == kategori), None)
    if not kayit:
        return

    _, ad = gorunum(kategori)
    st.caption(f"{ad} · {aylar} aylık bütçe")
    limit = st.number_input(
        f"{ad} için bütçe limiti (TL)", min_value=0, step=500,
        value=int(st.session_state.get(f"butce_{kategori}", 0)),
        key=f"butce_{kategori}", label_visibility="collapsed",
    )
    durum = butce_durumu(kayit["tutar"], limit)
    if not durum:
        st.caption("Limit girilirse doluluk burada görünür.")
        return

    ton = renk["artis"] if durum["asildi"] else renk["seri"]
    st.html(f"""
      <div class="blok" style="padding:14px 16px">
        <div class="satir-cubuk" style="height:10px">
          <div class="satir-dolgu" style="height:10px;width:{durum['genislik']}%;
               background:{ton}"></div>
        </div>
        <div class="hero-alt" style="margin-top:8px">
          {tl(kayit['tutar'])} / {tl(limit)} · %{durum['oran'] * 100:.0f}
          {'— <b>limit aşıldı</b>' if durum['asildi'] else f"· kalan {tl(durum['kalan'])}"}
        </div>
      </div>
    """)


def main() -> None:
    st.set_page_config(page_title="Harcama Analizi ve Kampanya Öneri Agent Sistemi",
                       page_icon="💳", layout="wide")
    renk = tema()
    st.html(stil(renk))

    with st.sidebar:
        st.subheader("Servis")
        st.session_state["api_kok"] = st.text_input("API adresi", VARSAYILAN_API)
        try:
            s = saglik(_kok())
            st.caption(f"{'✅' if s['status'] == 'ok' else '⚠️'} {s['status']} · {s['llm']}")
        except Exception:
            st.error("Servise ulaşılamıyor.\n\n`uvicorn src.api.main:app`", icon="🔌")
            st.stop()

        st.subheader("Müşteri")
        yalniz_izinli = st.checkbox("Yalnız pazarlama izni olanlar", value=False)
        kimlikler = [m["customer_id"] for m in musteri_listesi(_kok(), yalniz_izinli)]

        if st.button("Rastgele müşteri seç ", width="stretch"):
            import random

            st.session_state["musteri"] = random.choice(kimlikler)

        varsayilan = st.session_state.get("musteri", kimlikler[0])
        secim = st.selectbox(
            "Müşteri", options=kimlikler,
            index=kimlikler.index(varsayilan) if varsayilan in kimlikler else 0,
        )
        st.session_state["musteri"] = secim

        aylar = st.segmented_control("Dönem", [3, 6, 12], default=3,
                                     format_func=lambda a: f"{a} ay") or 3

    # Başlık ve alt metin README'deki ürün tanımıyla aynı sözü veriyor: iki
    # etkileşim yolu (dashboard + sohbet) ve teklifin analizin üstüne oturması.
    st.html(
        '<div class="marka-serit">'
        "<h1>Harcama Analizi ve Kampanya Öneri Agent Sistemi</h1>"
        "<p>Paranızın nereye gittiğini görün, sorularınızı sorun — "
        "size uygun kampanya, harcamanızdan çıkan gerekçesiyle birlikte gelsin.</p>"
        "</div>"
    )

    try:
        p = profil(_kok(), secim)
        analiz = harcama(_kok(), secim, aylar)
    except RuntimeError as hata:
        st.error(str(hata), icon="🔌")
        st.stop()

    if not p.get("opt_in_marketing"):
        # Ekranda yalnız durum yazıyor; kuralın kod tarafındaki karşılığı
        # (`check_eligibility`'nin ilk kuralda durması) geliştirici bilgisidir.
        st.warning("Bu müşterinin pazarlama izni yok.")

    analiz_sekmesi, ajan_sekmesi = st.tabs(["Harcama Analizi", "Agent'a sor"])
    with analiz_sekmesi:
        st.caption(
            f"{secim} · {p.get('customer_segment', '-')} segment · {p.get('age', '-')} yaş · "
            f"kıdem {p.get('tenure_months', '-')} ay · "
            f"pazarlama izni {'var' if p.get('opt_in_marketing') else 'YOK'}"
        )
        if p.get("behavior_segment"):
            # Bankanın ticari segmenti yukarıda; bu, harcama davranışından
            # KMeans ile TÜRETİLEN segment. Projenin varlık sebebi bu ayrım.
            st.html(
                f'<span class="rozet" style="background:{renk["marka"]};'
                f'color:{renk["marka_ink"]}">🧭 {html.escape(p["behavior_segment"])}</span>'
                f'<span style="color:{renk["ikincil"]};font-size:.82rem">'
                "&nbsp; davranışsal segment — harcamadan türetildi</span>"
            )
        analiz_goster(analiz, aylar, renk)
    with ajan_sekmesi:
        sohbet_goster(secim, analiz)


if __name__ == "__main__":
    main()
