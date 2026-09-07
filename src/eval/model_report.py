"""Model değerlendirme raporu — kaydedilmiş paketlerden, yeniden eğitmeden.

    python -m src.eval.model_report
    python -m src.eval.model_report --out reports

Metrikler zaten eğitim sırasında hesaplanıp `.joblib` paketlerinin içine
yazılıyor; buradaki iş onları rapora çevirmek. Yeniden eğitmiyoruz: rapor,
**dosyadaki modelin** — yani serviste çalışan modelin — karnesi olmalı, o an
eğitilen başka bir modelin değil.

İki şey bilerek raporda tutuluyor:

  - **Baseline tablosu eksiksiz.** Tek satırlık kategori kuralı PR-AUC'de modeli
    yeniyor. Bunu tablodan çıkarmak raporu güzelleştirir ve değersizleştirir.
  - **Elenen k değerleri.** Segmentasyonda en yüksek silhouette'i alan k, en
    küçük kümesi nüfusun %5'inin altında kaldığı için elenmiş olabilir; rapor
    hangi k'nın neden elendiğini göstermezse "neden 5 değil de 3?" sorusunun
    cevabı kaybolur.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.models.segmentation import MIN_KUME_ORANI


def _yuzde(deger: float) -> str:
    return f"%{deger * 100:.1f}"


def propensity_bolumu(paket: dict) -> list[str]:
    m = paket["metrics"]
    sizinti = paket["leakage_check"]

    satirlar = [
        "## Kampanya kabul eğilimi modeli",
        "",
        f"- Algoritma: **{paket['algorithm']}** · `{paket['params']}`",
        f"- Eğitim: {paket['n_rows']:,} (müşteri, kampanya) çifti · "
        f"{paket['n_customers']:,} müşteri · {len(paket['feature_columns'])} özellik",
        f"- Kabul oranı: {_yuzde(paket['acceptance_rate'])} · eğitim tarihi "
        f"{paket['trained_at']}",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
        f"| AUC | **{m['auc']:.3f}** |",
        f"| PR-AUC | {m['pr_auc']:.3f} |",
        f"| precision@1 | {m['precision_at_1']:.3f} ({m['n_ranked_customers']} müşteri) |",
        f"| Rastgeleye karşı kazanç | {m['uplift_vs_random']:.2f}× |",
        "",
        "### Baseline karşılaştırması",
        "",
        "| Yöntem | AUC | PR-AUC | precision@1 |",
        "|---|---:|---:|---:|",
    ]
    for ad, b in paket["baselines"].items():
        satirlar.append(
            f"| {ad} | {b['auc']:.3f} | {b['pr_auc']:.3f} | {b['precision_at_1']:.3f} |"
        )
    satirlar.append(
        f"| **model** | **{m['auc']:.3f}** | **{m['pr_auc']:.3f}** | "
        f"**{m['precision_at_1']:.3f}** |"
    )

    kural = paket["baselines"].get("kategori_kurali")
    if kural and kural["pr_auc"] > m["pr_auc"]:
        satirlar += [
            "",
            f"Tek satırlık kategori kuralı PR-AUC'de modeli yeniyor "
            f"({kural['pr_auc']:.3f} > {m['pr_auc']:.3f}). Model AUC ve precision@1'de "
            "önde. Bu satır raporda bilerek duruyor: baseline'ı gizlemek modeli "
            "olduğundan iyi gösterir.",
        ]

    satirlar += [
        "",
        "### Sızıntı kontrolü",
        "",
        f"- Müşteri-gruplu bölme (raporlanan): **{sizinti['grouped_auc']:.4f}**",
        f"- Rastgele satır bölmesi: {sizinti['random_split_auc']:.4f} "
        f"({sizinti['shared_customers']} müşteri hem eğitimde hem testte)",
        f"- Fark: **{sizinti['gap']:+.4f}** · gözlemlenebilir tavan "
        f"{sizinti['observable_ceiling']}",
        "",
        "Bir müşterinin tüm teklifleri aynı müşteri bloğunu taşır; satır bazlı bölme "
        "modelin kişiyi tanıyıp üretecin sakladığı gizli eğilimi ezberlemesine izin "
        "verir. Fark sıfıra yakınsa model kişiyi değil deseni öğrenmiştir.",
        "",
    ]
    return satirlar


def segmentasyon_bolumu(paket: dict, segmentler: pd.DataFrame | None) -> list[str]:
    satirlar = [
        "## Davranışsal segmentasyon",
        "",
        f"- Seçilen k: **{paket['chosen_k']}** · PCA {paket['n_components']} bileşen "
        f"(açıklanan varyans {_yuzde(paket['explained_variance'])})",
        f"- Davranışsal özellik: {len(paket['features'])} · eğitim tarihi "
        f"{paket['trained_at']}",
        "",
        "### k seçimi",
        "",
        "| k | Silhouette | En küçük küme | Sonuç |",
        "|---:|---:|---:|---|",
    ]
    en_yuksek = max(paket["silhouette_scores"], key=paket["silhouette_scores"].get)
    for k, skor in paket["silhouette_scores"].items():
        oran = paket["min_cluster_ratio"][k]
        if k == paket["chosen_k"]:
            sonuc = "**seçildi**"
        elif oran < MIN_KUME_ORANI:
            sonuc = f"elendi (< {_yuzde(MIN_KUME_ORANI)})"
        else:
            sonuc = "—"
        satirlar.append(f"| {k} | {skor:.3f} | {_yuzde(oran)} | {sonuc} |")

    if paket["min_cluster_ratio"][en_yuksek] < MIN_KUME_ORANI:
        satirlar += [
            "",
            f"En yüksek silhouette k={en_yuksek}'te ({paket['silhouette_scores'][en_yuksek]:.3f}) "
            f"ama en küçük kümesi nüfusun {_yuzde(paket['min_cluster_ratio'][en_yuksek])}'i — "
            "kampanya kurgulanamayacak bir artık. Segment bir hedef kitledir; iş kısıtı "
            "silhouette'ten önce gelir.",
        ]

    satirlar += ["", "### Bulunan segmentler", "", "| Küme | Ad | Müşteri |", "|---:|---|---:|"]
    for kume, ad in paket["cluster_names"].items():
        adet = (int((segmentler["behavior_cluster"] == kume).sum())
                if segmentler is not None else "-")
        pay = (f" ({adet / len(segmentler) * 100:.0f}%)"
               if segmentler is not None else "")
        satirlar.append(f"| {kume} | {ad} | {adet:,}{pay} |" if segmentler is not None
                        else f"| {kume} | {ad} | - |")
    satirlar.append("")
    return satirlar


def rapor_uret(config: dict) -> str:
    model_dir = resolve_path(config["models"]["dir"])
    islenmis = resolve_path(config["data"]["processed_dir"])

    propensity = joblib.load(model_dir / "propensity.joblib")
    segmentasyon = joblib.load(model_dir / "segmentation.joblib")
    segment_yolu = islenmis / "customer_segments.parquet"
    segmentler = pd.read_parquet(segment_yolu) if segment_yolu.exists() else None

    satirlar = ["# Model Değerlendirme Raporu", ""]
    satirlar += propensity_bolumu(propensity)
    satirlar += segmentasyon_bolumu(segmentasyon, segmentler)
    return "\n".join(satirlar)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Kaydedilmiş modellerden değerlendirme raporu üretir.")
    parser.add_argument("--out", type=str, default="reports",
                        help="rapor dizini ('' verilirse yalnız ekrana basar)")
    args = parser.parse_args(argv)

    config = load_config()
    try:
        rapor = rapor_uret(config)
    except FileNotFoundError as hata:
        print(f"HATA: {hata}\nÖnce modelleri eğitin: ./scripts/setup.sh", file=sys.stderr)
        return 1

    print(rapor)
    if args.out:
        out_dir = resolve_path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        yol = out_dir / "model_report.md"
        yol.write_text(rapor + "\n", encoding="utf-8")
        print(f"\nyazıldı: {yol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
