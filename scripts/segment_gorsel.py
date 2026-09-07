"""Segment profillerini sunum için görselleştirir.

    python scripts/segment_gorsel.py            # -> reports/segmentler.png

PCA saçılım grafiği yerine kategori profili çiziliyor: "bileşen 1 / bileşen 2"
eksenleri teknik olmayan dinleyiciye hiçbir şey anlatmaz, oysa "bu grup
harcamasının %31'ini seyahate ayırıyor" doğrudan anlaşılır ve segmentlerin
gerçekten farklı olduğunu kanıtlar.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, resolve_path
from src.features.categories import CATEGORIES

KATEGORI_ADI = {
    "market": "Market", "akaryakit": "Akaryakıt", "restoran": "Restoran",
    "giyim": "Giyim", "elektronik": "Elektronik", "seyahat": "Seyahat",
    "saglik": "Sağlık", "egitim": "Eğitim", "eglence": "Eğlence",
    "telekom": "Telekom", "online_alisveris": "Online alışveriş", "diger": "Diğer",
}
RENKLER = ["#B58900", "#1F6F5C", "#3E5060"]


def segment_renkleri(adlar) -> dict[str, str]:
    """Segment adı -> renk. İki grafikte aynı segment aynı renkte olmalı.

    Kümeye sırasına göre renk vermek, aynı segmenti bir grafikte yeşil diğerinde
    lacivert yapıyordu; renk kimliği takip eder."""
    return {ad: RENKLER[i % len(RENKLER)] for i, ad in enumerate(sorted(adlar))}


def dagilim_grafigi(cfg: dict) -> Path:
    """Klasik KMeans görseli: müşteriler 2 boyuta indirgenip kümeye göre boyanır.

    Yedek slayt için. Ana slaytta kategori profili tercih edildi çünkü "bileşen 1"
    ekseni teknik olmayan dinleyiciye hiçbir şey anlatmaz; buradaki grafiğin
    söylediği tek şey "kümeler gerçekten ayrışıyor" — teknik soru gelirse değerli.
    """
    import joblib

    islenmis = resolve_path(cfg["data"]["processed_dir"])
    paket = joblib.load(resolve_path(cfg["models"]["dir"]) / "segmentation.joblib")
    ozellik = pd.read_parquet(islenmis / "customer_features.parquet")
    segment = pd.read_parquet(islenmis / "customer_segments.parquet")
    df = ozellik.merge(segment, on="customer_id")

    X = paket["pipeline"].transform(df[paket["features"]].to_numpy(dtype=float))
    merkez = paket["kmeans"].cluster_centers_

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    renk_haritasi = segment_renkleri(paket["cluster_names"].values())
    for kume, ad in sorted(paket["cluster_names"].items(), key=lambda x: x[1]):
        maske = df["behavior_cluster"] == kume
        ax.scatter(X[maske, 0], X[maske, 1], s=12, alpha=0.45,
                   color=renk_haritasi[ad], label=f"{ad} ({maske.sum():,})")
        ax.scatter(merkez[kume, 0], merkez[kume, 1], marker="X", s=220,
                   color=renk_haritasi[ad], edgecolor="white", linewidth=2)

    ax.set_xlabel("1. bileşen — harcama deseninin en ayırt edici yönü")
    ax.set_ylabel("2. bileşen")
    ax.set_title("Müşteriler harcama davranışına göre ayrışıyor",
                 loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    cikti = resolve_path("reports") / "segmentler_dagilim.png"
    fig.savefig(cikti, dpi=160, bbox_inches="tight", facecolor="white")
    return cikti


def main() -> int:
    cfg = load_config()
    islenmis = resolve_path(cfg["data"]["processed_dir"])
    ozellik = pd.read_parquet(islenmis / "customer_features.parquet")
    segment = pd.read_parquet(islenmis / "customer_segments.parquet")

    df = ozellik.merge(segment, on="customer_id")
    pay_kolonlari = [f"cat_share_{k}" for k in CATEGORIES if f"cat_share_{k}" in df.columns]
    profil = df.groupby("behavior_segment")[pay_kolonlari + ["monetary_total"]].mean()
    boyut = df["behavior_segment"].value_counts()
    nufus = df[pay_kolonlari].mean()

    # Segmentleri en ayırt edici kategorilere göre sırala (lift = küme payı / nüfus payı)
    lift = profil[pay_kolonlari].div(nufus, axis=1)
    onemli = lift.max(axis=0).sort_values(ascending=False).head(6).index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    segmentler = list(profil.index)
    n, genislik = len(onemli), 0.26
    konum = range(n)

    renk_haritasi = segment_renkleri(segmentler)
    for i, seg in enumerate(segmentler):
        ax1.bar([k + (i - 1) * genislik for k in konum],
                [profil.loc[seg, c] * 100 for c in onemli],
                width=genislik, label=f"{seg} ({boyut[seg]:,} kişi)",
                color=renk_haritasi[seg])

    ax1.set_xticks(list(konum))
    ax1.set_xticklabels([KATEGORI_ADI.get(c.replace("cat_share_", ""), c) for c in onemli])
    ax1.set_ylabel("harcama payı (%)")
    ax1.set_title("Segmentler harcamasını nereye ayırıyor?", loc="left",
                  fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.25)

    ax2.barh(segmentler, [profil.loc[s, "monetary_total"] / 1000 for s in segmentler],
             color=[renk_haritasi[s] for s in segmentler], height=0.5)
    ax2.set_xlabel("yıllık ortalama harcama (bin TL)")
    ax2.set_title("Harcama düzeyi", loc="left", fontsize=13, fontweight="bold")
    ax2.set_yticklabels([s.replace(" ", "\n", 1) for s in segmentler], fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    cikti = resolve_path("reports") / "segmentler.png"
    cikti.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"yazıldı: {cikti}")
    print(f"yazıldı: {dagilim_grafigi(cfg)}")
    for seg in segmentler:
        print(f"  {seg:38} {boyut[seg]:>5,} kişi  "
              f"ort. {profil.loc[seg, 'monetary_total']:>10,.0f} TL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
