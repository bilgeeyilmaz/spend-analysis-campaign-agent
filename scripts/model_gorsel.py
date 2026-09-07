"""Kampanya kabul modelini sunumda anlatan görseller.

    python -m scripts.model_gorsel      # -> reports/model_isabet.png, model_onem.png

Soyut metrik yerine iş diline çevrilmiş iki grafik:
  1. "100 teklifin kaçı kabul edilir" — yöntem karşılaştırması
  2. Modelin en çok önemsediği bilgiler (özellik önemi)
"""

from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, resolve_path

RENK_MODEL, RENK_DIGER = "#B58900", "#8A8A80"

#: Özellik adı -> sunumda okunacak Türkçe karşılık. Ham ad ("fit_category_share")
#: teknik olmayan dinleyiciye hiçbir şey anlatmaz.
OZELLIK_ADI = {
    "fit_category_share": "Kampanya kategorisinin müşteri harcamasındaki payı",
    "fit_monthly_amount": "O kategorideki aylık harcama tutarı",
    "budget_ratio": "Kampanyanın kalan bütçesi",
    "min_spend_friction": "Asgari harcama şartının zorluğu",
    "reward_value": "Kampanyanın ödül oranı",
    "monetary_total": "Toplam harcama",
    "recency_days": "Son işlemden bu yana geçen gün",
    "spend_momentum": "Harcama ivmesi",
    "installment_ratio": "Taksit kullanma oranı",
    "active_months": "Aktif ay sayısı",
}


#: Kategori kimliği -> Türkçe ad. Kimlikler ASCII (`egitim`, `online_alisveris`);
#: doğrudan etikete çevrilirse ekranda "egitim", "alisveris" diye yazım hatası
#: görünür.
KATEGORI_ADI = {
    "market": "market", "akaryakit": "akaryakıt", "restoran": "restoran",
    "giyim": "giyim", "elektronik": "elektronik", "seyahat": "seyahat",
    "saglik": "sağlık", "egitim": "eğitim", "eglence": "eğlence",
    "telekom": "telekom", "online_alisveris": "online alışveriş", "diger": "diğer",
}


def _kategori(kimlik: str) -> str:
    return KATEGORI_ADI.get(kimlik, kimlik.replace("_", " "))


def _okunur(ad: str) -> str:
    if ad in OZELLIK_ADI:
        return OZELLIK_ADI[ad]
    if ad.startswith("cat_trend_"):
        return f"{_kategori(ad.removeprefix('cat_trend_')).capitalize()} harcama eğilimi"
    if ad.startswith("cat_monthly_"):
        return f"Aylık {_kategori(ad.removeprefix('cat_monthly_'))} harcaması"
    if ad.startswith("cat_share_"):
        return f"{_kategori(ad.removeprefix('cat_share_')).capitalize()} payı"
    return ad.replace("_", " ")


ETIKET = {
    "rastgele": "Rastgele seçim",
    "kampanya_populerligi": "En popüler kampanya",
    "kategori_kurali": "Basit kategori kuralı",
}


def isabet_grafigi(paket: dict) -> "Path":
    """İlk sıradaki önerinin kabul oranı — yöntem karşılaştırması."""
    satirlar = [(ETIKET.get(ad, ad), b["precision_at_1"] * 100)
                for ad, b in paket["baselines"].items()]
    satirlar.append(("Model (LightGBM)", paket["metrics"]["precision_at_1"] * 100))

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    adlar = [a for a, _ in satirlar]
    degerler = [d for _, d in satirlar]
    renkler = [RENK_MODEL if a.startswith("Model") else RENK_DIGER for a in adlar]

    cubuklar = ax.barh(adlar, degerler, color=renkler, height=0.6)
    for cubuk, deger in zip(cubuklar, degerler):
        ax.text(deger + 1, cubuk.get_y() + cubuk.get_height() / 2,
                f"{deger:.0f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("ilk sırada önerilen kampanyanın kabul edilme oranı (%)")
    ax.set_title("100 müşteriye ilk önerimizi sunsak, kaçı kabul eder?",
                 loc="left", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(degerler) * 1.25)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    yol = resolve_path("reports") / "model_isabet.png"
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor="white")
    return yol


def onem_grafigi(paket: dict, n: int = 8) -> "Path | None":
    """Modelin en çok kullandığı bilgiler. LightGBM'in kendi 'gain' ölçüsü."""
    model = paket["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    onem = pd.Series(model.feature_importances_, index=paket["feature_columns"])
    onem = onem.sort_values(ascending=False).head(n)[::-1]

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.barh([_okunur(a) for a in onem.index], onem.to_numpy(),
            color=RENK_MODEL, height=0.62)
    ax.set_xlabel("modelin karar verirken kullanma sıklığı")
    ax.set_title("Model en çok neye bakıyor?", loc="left", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    yol = resolve_path("reports") / "model_onem.png"
    fig.savefig(yol, dpi=160, bbox_inches="tight", facecolor="white")
    return yol


def main() -> int:
    cfg = load_config()
    paket = joblib.load(resolve_path(cfg["models"]["dir"]) / "propensity.joblib")
    print("yazıldı:", isabet_grafigi(paket))
    yol = onem_grafigi(paket)
    print("yazıldı:", yol) if yol else print("özellik önemi bu modelde yok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
