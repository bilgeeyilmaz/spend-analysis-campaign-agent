"""Davranışsal müşteri segmentasyonu (KMeans).

    python -m src.models.segmentation
    python -m src.models.segmentation --check-personas   # sadece sentetik veride

Bankanın zaten bir segmenti var (`Customer.customer_segment`: mass/affluent/private)
ve o **ticari** bir segmenttir — bakiye ve gelire dayanır. Buradaki segment
**davranışsaldır**: müşteri parasını nereye, hangi kanaldan, ne sıklıkla harcıyor.
İki affluent müşteriden biri seyahat ediyor, diğeri market alışverişi yapıyorsa
aynı kampanyayı almamalılar — projenin varlık sebebi bu ayrımdır.

Hat: davranışsal özellikler -> StandardScaler -> PCA -> KMeans (k silhouette ile).

İsimlendirme otomatiktir: küme merkezleri nüfus ortalamasıyla karşılaştırılır ve
en ayırt edici kategori + harcama düzeyi + dijitallik üzerinden Türkçe bir etiket
üretilir. Elle isim vermiyoruz — veri değişince isim de değişsin, sunumda
"bu isimleri ben uydurdum" durumuna düşmeyelim.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config, resolve_path
from src.features.build import BEHAVIORAL_FEATURES

#: Kategori kolonu -> Türkçe segment ifadesi. İsimlendirme sözlüğü.
KATEGORI_ETIKET: dict[str, str] = {
    "market": "market odaklı",
    "akaryakit": "akaryakıt ağırlıklı",
    "restoran": "dışarıda yemek ağırlıklı",
    "giyim": "giyim odaklı",
    "elektronik": "elektronik ağırlıklı",
    "seyahat": "seyahat eden",
    "saglik": "sağlık harcamalı",
    "egitim": "eğitim harcamalı",
    "eglence": "eğlence ve abonelik odaklı",
    "telekom": "fatura ağırlıklı",
    "online_alisveris": "dijital alışverişçi",
    "diger": "karma harcamalı",
}

#: Bir kategorinin isme girmesi için kümedeki payının nüfus payına oranı (lift)
#: en az bu kadar olmalı. 1.35 = "nüfus ortalamasının %35 üstünde".
LIFT_ESIK = 1.35

#: Bir küme nüfusun bu oranından küçükse o k değeri reddedilir. Segment bir
#: KAMPANYA HEDEF KİTLESİDİR; 20 kişilik bir kümeye kampanya kurgulanmaz ve
#: demoda "modelin bulduğu segment" diye gösterilemez.
MIN_KUME_ORANI = 0.05


def _harcama_duzeyi(kume_ort: float, dusuk: float, yuksek: float) -> str:
    if kume_ort >= yuksek:
        return "yüksek harcamalı"
    if kume_ort <= dusuk:
        return "tutumlu"
    return "orta segment"


def isimlendir(profil: pd.DataFrame, nufus: pd.Series, harcama: pd.Series) -> dict[int, str]:
    """Küme merkezlerinden okunabilir Türkçe segment adları üretir.

    `profil`  : küme × özellik ortalamaları (orijinal birimlerde)
    `nufus`   : tüm nüfusun aynı özelliklerdeki ortalaması
    `harcama` : tüm müşterilerin `monetary_total` serisi (tercil için)

    Ayırt ediciliği **lift** ile ölçüyoruz: kümenin kategori payı / nüfusun aynı
    kategorideki payı. Küme merkezlerinin standart sapmasına bakmak yanıltıcıydı —
    5 küme, 5 gözlem demek; market gibi herkeste yüksek olan bir kategori küçük
    bir mutlak farkla "ayırt edici" çıkabiliyordu. Lift ile "bu kümede seyahat
    payı nüfusun 2,4 katı" gibi doğrudan savunulabilir bir ifade elde ediyoruz.

    Harcama düzeyi de NÜFUS tercillerine göre belirlenir; küme ortalamalarının
    kendi içindeki tercili, 180 bin TL'lik bir kümeyi 670 binlik bir kümeyle aynı
    anda "yüksek harcamalı" yapıyordu.
    """
    pay_kolonlari = [c for c in profil.columns if c.startswith("cat_share_")]
    lift = profil[pay_kolonlari].div(nufus[pay_kolonlari].replace(0, np.nan), axis=1)

    dusuk, yuksek = harcama.quantile([0.33, 0.67])

    isimler: dict[int, str] = {}
    kullanilan: set[str] = set()
    for kume in profil.index:
        sirali = lift.loc[kume].sort_values(ascending=False)
        adaylar = [
            KATEGORI_ETIKET[c.replace("cat_share_", "")]
            for c in sirali.index if sirali[c] >= LIFT_ESIK
        ] or ["dengeli harcayan"]

        duzey = _harcama_duzeyi(profil.loc[kume, "monetary_total"], dusuk, yuksek)
        ad = f"{duzey} {adaylar[0]}"
        if ad in kullanilan and len(adaylar) > 1:
            ad = f"{duzey} {adaylar[0]} / {adaylar[1]}"
        while ad in kullanilan:
            ad += " (2)"
        kullanilan.add(ad)
        isimler[kume] = ad
    return isimler


def en_iyi_k(
    X: np.ndarray, k_range: list[int], random_state: int, ornek: int
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Silhouette skoruna göre küme sayısını seçer.

    Ek kısıt: en küçük küme nüfusun %5'inden küçükse o k elenir. Salt silhouette
    ile seçim, 20 kişilik bir artık kümeyi barındıran bir k'yı seçebiliyor
    (k=5'te 23 kişilik bir küme çıkmıştı) — matematiksel olarak geçerli, iş
    olarak kullanılamaz bir sonuç.
    """
    skorlar: dict[int, float] = {}
    en_kucuk: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiket = km.fit_predict(X)
        skorlar[k] = float(silhouette_score(
            X, etiket, sample_size=min(ornek, len(X)), random_state=random_state
        ))
        en_kucuk[k] = float(np.bincount(etiket).min() / len(etiket))

    uygun = {k: s for k, s in skorlar.items() if en_kucuk[k] >= MIN_KUME_ORANI}
    if not uygun:
        raise ValueError(
            f"Hiçbir k, en küçük küme >= %{MIN_KUME_ORANI:.0%} şartını sağlamadı. "
            f"k_range'i küçült veya MIN_KUME_ORANI'nı gözden geçir. "
            f"En küçük küme oranları: { {k: round(v,3) for k,v in en_kucuk.items()} }"
        )
    return max(uygun, key=uygun.get), skorlar, en_kucuk


def segment_et(tablo: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Özellik tablosunu segmentlere ayırır; (sonuç tablosu, model paketi) döner."""
    cfg = config["models"]["segmentation"]
    X_ham = tablo[BEHAVIORAL_FEATURES].to_numpy(dtype=float)

    on_isleme = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=cfg["pca_variance"], random_state=cfg["random_state"])),
    ])
    X = on_isleme.fit_transform(X_ham)

    k, skorlar, en_kucuk = en_iyi_k(
        X, cfg["k_range"], cfg["random_state"], cfg["silhouette_sample"]
    )
    kmeans = KMeans(n_clusters=k, random_state=cfg["random_state"], n_init=10)
    etiketler = kmeans.fit_predict(X)

    sonuc = tablo[["customer_id"]].copy()
    sonuc["behavior_cluster"] = etiketler

    # Profil: küme ortalamaları ORİJİNAL birimlerde (PCA uzayında değil) —
    # isimlendirme ve sunum grafikleri bunu okur.
    profil_kolonlari = BEHAVIORAL_FEATURES + ["monetary_total", "avg_ticket", "age"]
    profil_kolonlari = list(dict.fromkeys(c for c in profil_kolonlari if c in tablo.columns))
    profil = tablo.assign(behavior_cluster=etiketler).groupby("behavior_cluster")[
        profil_kolonlari
    ].mean()

    isimler = isimlendir(profil, tablo[profil_kolonlari].mean(), tablo["monetary_total"])
    sonuc["behavior_segment"] = sonuc["behavior_cluster"].map(isimler)

    paket = {
        "pipeline": on_isleme,
        "kmeans": kmeans,
        "features": BEHAVIORAL_FEATURES,
        "cluster_names": isimler,
        "silhouette_scores": skorlar,
        "min_cluster_ratio": en_kucuk,
        "chosen_k": k,
        "n_components": int(on_isleme.named_steps["pca"].n_components_),
        "explained_variance": float(on_isleme.named_steps["pca"].explained_variance_ratio_.sum()),
        "trained_at": date.today().isoformat(),
    }
    return sonuc, paket, profil


def persona_kontrolu(sonuc: pd.DataFrame, config: dict) -> pd.DataFrame | None:
    """SADECE SENTETİK VERİDE: kümeler gizli persona'ları geri buldu mu?

    Üreteç aynı tohumla (seed) yeniden çalıştırılıp persona'lar bellekte elde
    edilir — veriye hiç yazılmadıkları için modelin onları görmesi mümkün değildi.
    Bu, "model gizli yapıyı geri buldu" iddiasının ölçülebilir kanıtıdır.

    Gerçek veriye geçildiğinde bu fonksiyon anlamsızlaşır ve çağrılmaz.
    """
    if config["data"]["source"] != "json":
        return None

    from src.data.generator import generate_customers  # yalnız sentetik yolda

    gen = config["generator"]
    rng = np.random.default_rng(gen["seed"])
    musteriler, latent = generate_customers(
        rng, gen["n_customers"], date.today(), gen["opt_in_rate"]
    )
    persona = pd.DataFrame({
        "customer_id": [c["customer_id"] for c in musteriler],
        "persona": [latent[c["customer_id"]]["persona"] for c in musteriler],
    })
    birlesik = sonuc.merge(persona, on="customer_id", how="inner")
    if birlesik.empty:
        return None
    return pd.crosstab(birlesik["behavior_segment"], birlesik["persona"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Davranışsal segmentasyon (KMeans).")
    parser.add_argument("--check-personas", action="store_true",
                        help="kümeleri üretecin gizli persona'larıyla karşılaştır (sentetik veri)")
    args = parser.parse_args(argv)

    config = load_config()
    features_path = resolve_path(config["data"]["processed_dir"]) / "customer_features.parquet"
    if not features_path.exists():
        print(f"HATA: {features_path} yok. Önce: python -m src.features.build", file=sys.stderr)
        return 1
    tablo = pd.read_parquet(features_path)

    sonuc, paket, profil = segment_et(tablo, config)

    model_dir = resolve_path(config["models"]["dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(paket, model_dir / "segmentation.joblib")
    out_path = resolve_path(config["data"]["processed_dir"]) / "customer_segments.parquet"
    sonuc.to_parquet(out_path, index=False)

    print("\n=== Davranışsal Segmentasyon ===\n")
    print(f"  davranışsal özellik : {len(BEHAVIORAL_FEATURES)}")
    print(f"  PCA bileşeni        : {paket['n_components']} "
          f"(açıklanan varyans {paket['explained_variance']:.1%})")
    print(f"  seçilen k           : {paket['chosen_k']}")
    print("\n--- Silhouette skorları ---")
    for k, skor in paket["silhouette_scores"].items():
        oran = paket["min_cluster_ratio"][k]
        if k == paket["chosen_k"]:
            isaret = "  <-- seçildi"
        elif oran < MIN_KUME_ORANI:
            isaret = f"  (elendi: en küçük küme %{oran*100:.1f})"
        else:
            isaret = ""
        print(f"  k={k}  silhouette {skor:.3f}   en küçük küme %{oran*100:>4.1f}{isaret}")

    print("\n--- Segment profilleri ---")
    dagilim = sonuc["behavior_segment"].value_counts()
    for kume, ad in paket["cluster_names"].items():
        n = int((sonuc["behavior_cluster"] == kume).sum())
        satir = profil.loc[kume]
        paylar = satir[[c for c in profil.columns if c.startswith("cat_share_")]]
        ilk_uc = paylar.sort_values(ascending=False).head(3)
        print(f"\n  [{kume}] {ad}")
        print(f"      {n:,} müşteri ({n/len(sonuc):.1%})   "
              f"yıllık harcama ort {satir['monetary_total']:,.0f} TL   "
              f"ort sepet {satir['avg_ticket']:,.0f} TL")
        print("      baskın kategoriler: " + ", ".join(
            f"{c.replace('cat_share_','')} %{v*100:.0f}" for c, v in ilk_uc.items()
        ))
        print(f"      online payı %{satir['channel_share_online']*100:.0f}   "
              f"taksitli işlem %{satir['installment_ratio']*100:.0f}   "
              f"aktif ay {satir['active_months']:.1f}")

    if args.check_personas:
        tablo_persona = persona_kontrolu(sonuc, config)
        if tablo_persona is None:
            print("\n[!] Persona kontrolü yalnız sentetik (json) kaynakta çalışır.")
        else:
            print("\n--- Gizli persona × bulunan segment ---")
            print("    (persona veriye hiç yazılmadı; model onu göremezdi)\n")
            oran = tablo_persona.div(tablo_persona.sum(axis=0), axis=1)
            print(oran.round(2).to_string())
            baskin = oran.max(axis=0).mean()
            print(f"\n  Her persona'nın en yoğun segmentte toplanma oranı (ort): {baskin:.0%}")

    print(f"\n  yazıldı: {out_path}")
    print(f"  yazıldı: {model_dir / 'segmentation.joblib'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
