"""Guardrail akışını sunum için çizer.

    python -m scripts.guardrail_gorsel      # -> reports/guardrails.png

Anlatılmak istenen tek şey var: LLM'in yazdığı metin müşteriye DOĞRUDAN gitmiyor.
Çizim bu yüzden denetimi yolun ortasına, tek kapı olarak koyuyor.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import resolve_path

ALTIN, KOYU, GRI, YESIL, KIRMIZI = "#B58900", "#12120F", "#6B6B63", "#1F6F5C", "#B0483A"


def kutu(ax, x, y, w, h, baslik, alt="", *, kenar=GRI, dolgu="#FFFFFF",
         baslik_pt=12, alt_pt=8.6, baslik_renk=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor=dolgu, edgecolor=kenar, linewidth=2.0))
    ax.text(x + w / 2, y + h - (0.30 if alt else h / 2 + 0.05), baslik,
            ha="center", va="center", fontsize=baslik_pt, fontweight="bold",
            color=baslik_renk or KOYU)
    if alt:
        ax.text(x + w / 2, y + h / 2 - 0.28, alt, ha="center", va="center",
                fontsize=alt_pt, color=GRI if dolgu == "#FFFFFF" else KOYU,
                linespacing=1.6)


def ok(ax, x1, y1, x2, y2, renk=GRI, stil="-|>", yay=0.0, etiket="", etiket_renk=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=stil,
                                 mutation_scale=15, linewidth=1.8, color=renk,
                                 connectionstyle=f"arc3,rad={yay}"))
    if etiket:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, etiket, ha="center",
                fontsize=9.5, color=etiket_renk or renk, fontweight="bold")


def akis_grafigi() -> "Path":
    """Girdi -> LLM -> çıktı hattı; standart "AI guardrail" şemasının bu projedeki karşılığı.

    Referans şemalarda iki guardrail vardır: girdi ve çıktı. Bu projede ikisi de
    var ama içerikleri farklı — girdi tarafında istek doğrulama, LLM'e giden tool
    çıktılarında PII maskeleme, çıktı tarafında dört denetim.
    """
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.0); ax.axis("off")

    # Sistemin sınırı
    ax.add_patch(FancyBboxPatch(
        (1.5, 1.6), 10.2, 2.9, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="none", edgecolor="#C9C8BC", linewidth=1.6))

    y, h = 2.35, 1.5
    kutu(ax, 2.0, y, 2.6, h, "Girdi kontrolü",
         "soru uzunluğu,\nmüşteri doğrulama", dolgu="#EDE3C4", kenar=ALTIN)
    kutu(ax, 5.4, y, 2.6, h, "Agent + LLM",
         "tool calling ile\nrakamları çeker", dolgu="#FFFFFF", kenar=GRI)
    kutu(ax, 8.8, y, 2.6, h, "Çıktı denetimi",
         "4 guardrail:\nsayı · ödül · tavsiye · kampanya",
         dolgu=ALTIN, kenar=ALTIN, alt_pt=8.0)

    # Yatay hat
    ax.text(0.15, 3.28, "Müşteri\nsorusu", fontsize=11.5, fontweight="bold",
            color=KOYU, va="center", linespacing=1.5)
    ok(ax, 1.35, 3.1, 1.95, 3.1)
    ok(ax, 4.65, 3.1, 5.35, 3.1)
    ok(ax, 8.05, 3.1, 8.75, 3.1)
    ok(ax, 11.45, 3.1, 12.15, 3.1)
    ax.text(12.3, 3.28, "Doğrulanmış\ncevap", fontsize=11.5, fontweight="bold",
            color=YESIL, va="center", linespacing=1.5)

    # Yukarıdan girenler
    ok(ax, 6.7, 5.35, 6.7, 3.92, GRI)
    ax.text(6.7, 5.55, "sistem promptu", ha="center", fontsize=10, color=GRI)
    ok(ax, 5.9, 5.35, 5.9, 3.92, GRI)
    ax.text(5.9, 5.9, "tool çıktıları\n(PII maskelenmiş)", ha="center",
            fontsize=9.5, color=GRI, linespacing=1.5)

    # Aşağı düşenler
    ok(ax, 3.3, 2.3, 3.3, 1.0, KIRMIZI)
    ax.text(3.3, 0.75, "reddedilen istek\n(geçersiz müşteri / çok uzun soru)",
            ha="center", fontsize=9.5, color=KIRMIZI, linespacing=1.5)
    ok(ax, 10.1, 2.3, 10.1, 1.0, KIRMIZI)
    ax.text(10.1, 0.75, "düzeltme talebi → şablon metin\n(doğrulanmamış rakam geçemez)",
            ha="center", fontsize=9.5, color=KIRMIZI, linespacing=1.5)

    # Başlık en üstte: okların arasında kalırsa çizgiler yazının üstünden geçiyordu.
    ax.text(0.15, 5.75, "Agent, LLM'e giden ve LLM'den çıkan her şeyin arasında durur",
            fontsize=12, color=KOYU, fontweight="bold")

    fig.tight_layout()
    yol = resolve_path("reports") / "guardrails_akis.png"
    fig.savefig(yol, dpi=170, bbox_inches="tight", facecolor="white")
    return yol


def main() -> int:
    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 6.6); ax.axis("off")

    kutu(ax, 0.2, 3.6, 2.5, 1.5, "Agent", "tool calling ile\nrakamları çeker,\nmetni yazar")

    # Denetim kutusu — başlık kutunun DIŞINDA, üstte (maddelerin üstüne binmesin)
    ax.add_patch(FancyBboxPatch(
        (3.4, 2.6, ), 3.6, 3.3, boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor=ALTIN, edgecolor=ALTIN, linewidth=2.0))
    ax.text(5.2, 6.05, "GUARDRAILS", ha="center", fontsize=13,
            fontweight="bold", color=KOYU)
    for i, (ad, ne) in enumerate([
        ("İzlenebilirlik", "her sayı bir tool çıktısına dayanmalı"),
        ("Ödül doğruluğu", "oran, önerilen kampanyanın mı?"),
        ("Tavsiye sınırı", "yatırım tavsiyesi yok"),
        ("Kampanya kontrolü", "katalog dışı kod geçemez"),
    ]):
        y = 5.45 - i * 0.78
        ax.text(3.62, y, f"{i + 1}.", fontsize=10, fontweight="bold", color=KOYU)
        ax.text(3.95, y, ad, fontsize=10.5, fontweight="bold", color=KOYU)
        ax.text(3.95, y - 0.26, ne, fontsize=8.2, color="#4A3B00")

    ok(ax, 2.75, 4.35, 3.35, 4.35)

    kutu(ax, 7.9, 4.6, 3.0, 1.2, "Geçti", "metin müşteriye gider",
         kenar=YESIL, baslik_renk=YESIL)
    kutu(ax, 7.9, 2.7, 3.0, 1.2, "Geçemedi", "bir düzeltme hakkı",
         kenar=KIRMIZI, baslik_renk=KIRMIZI)
    ok(ax, 7.05, 4.85, 7.85, 5.15, YESIL)
    ok(ax, 7.05, 3.65, 7.85, 3.35, KIRMIZI)

    kutu(ax, 11.4, 4.6, 2.9, 1.2, "Müşteri", "doğrulanmış metin",
         kenar=YESIL, baslik_renk=YESIL)
    ok(ax, 10.95, 5.2, 11.35, 5.2, YESIL)

    # Düzeltme geri dönüşü: kutuların ALTINDAN dolanıyor
    # Düzeltme talebi modele döner: ok AGENT kutusuna gider
    ok(ax, 8.4, 2.65, 1.45, 3.55, KIRMIZI, yay=-0.32)
    ax.text(4.9, 1.55, "modelden düzeltmesi istenir", ha="center",
            fontsize=9.5, color=KIRMIZI, fontweight="bold")

    # İkinci deneme de geçemezse şablon metne
    kutu(ax, 11.4, 1.6, 2.9, 1.3, "Şablon metin",
         "araç çıktılarından kurulur\n— uydurma yok", kenar=GRI)
    ok(ax, 10.95, 2.85, 11.35, 2.3, GRI)
    ax.text(10.6, 2.05, "ikinci denemede\nde geçemezse", ha="center", va="center",
            fontsize=8.8, color=GRI, style="italic", linespacing=1.5)
    ok(ax, 12.85, 2.95, 12.85, 4.55, GRI)

    ax.text(0.2, 0.85, "Her cevap — geçen de geçemeyen de — denetim kaydına yazılır.",
            fontsize=11, color=KOYU, fontweight="bold")
    ax.text(0.2, 0.45,
            "Ölçüldü: bir değerlendirme koşusunda 5 hatalı rakam yakalandı, hiçbiri müşteriye ulaşmadı.",
            fontsize=10.5, color=GRI)

    fig.tight_layout()
    yol = resolve_path("reports") / "guardrails.png"
    fig.savefig(yol, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"yazıldı: {yol}")
    print(f"yazıldı: {akis_grafigi()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
