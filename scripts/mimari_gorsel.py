"""Mimari şemasını sunum için çizer.

    python -m scripts.mimari_gorsel      # -> reports/mimari.png

Kutular iş diliyle adlandırıldı; teknoloji adları ikincil satırda küçük punto.
"Kurallar" kutusu bilerek farklı renkte: anlatının döndüğü nokta orası —
modelin puanı bu kapıyı geçemiyor.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import resolve_path

ALTIN, KOYU, GRI, BEYAZ = "#B58900", "#12120F", "#6B6B63", "#FFFFFF"

KUTULAR = [
    ("Veri", "kart işlemleri\nJSON · SQL", GRI),
    ("Harcama Özeti", "kategori, trend,\ndüzenli ödemeler", GRI),
    ("Tahmin Modeli", "kim hangi kampanyayı\nkabul eder?  ·  LightGBM", GRI),
    ("Kurallar", "izin, bütçe, tarih, yaş\n— kapı burada", ALTIN),
    ("Agent", "tool calling + guardrails\nLLM", GRI),
    ("Arayüz", "dashboard ve sohbet\nFastAPI · Streamlit", GRI),
]


def main() -> int:
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, len(KUTULAR) * 2.4)
    ax.set_ylim(0, 3)
    ax.axis("off")

    genislik, yukseklik = 2.0, 1.45
    for i, (baslik, alt, renk) in enumerate(KUTULAR):
        x = i * 2.4 + 0.1
        ax.add_patch(FancyBboxPatch(
            (x, 1.0), genislik, yukseklik,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=renk if renk == ALTIN else BEYAZ,
            edgecolor=renk, linewidth=2.2,
        ))
        ax.text(x + genislik / 2, 2.06, baslik, ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=KOYU if renk == ALTIN else KOYU)
        ax.text(x + genislik / 2, 1.52, alt, ha="center", va="center",
                fontsize=8.6, color=KOYU if renk == ALTIN else GRI, linespacing=1.5)

        if i < len(KUTULAR) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + genislik + 0.06, 1.72), (x + 2.4 + 0.04, 1.72),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=GRI,
            ))

    ax.text(0.1, 0.55,
            "Her katman yalnız bir öncekinin çıktısını kullanır. "
            "Soldaki kutu değişirse — yapay veri yerine gerçek banka verisi — "
            "sağdaki beşi aynı kalır.",
            fontsize=10.5, color=GRI, va="center")
    ax.text(0.1, 0.18,
            "Model sıralar, kurallar karar verir: kurala takılan kampanya agent'a hiç ulaşmaz.",
            fontsize=10.5, color=KOYU, va="center", fontweight="bold")

    fig.tight_layout()
    yol = resolve_path("reports") / "mimari.png"
    fig.savefig(yol, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"yazıldı: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
