"""Kampanya kabul eğilimi (propensity) modeli.

    python -m src.models.propensity
    python -m src.models.propensity --algorithm hist   # libomp yoksa
    python -m src.models.propensity --importance       # permütasyon önemleri (yavaş)

Girdi bir (müşteri, kampanya) ÇİFTİdir; hedef `Interaction.accepted`. Kampanya
kimliği özellik değildir, kampanya ÖZNİTELİKLERİ özelliktir — bu yüzden katalogda
yeni bir kampanya açıldığında model yeniden eğitilmez, yeni satırın öznitelikleri
skorlanır. Kampanya başına ayrı model kurmak bu özelliği yok eder.

Üç özellik bloğu:

  müşteri  : `features.build.model_feature_columns()` (70 kolon)
  kampanya : ödül tipi/değeri, min_spend, süre, kalan bütçe oranı, öncelik, kanal
  çapraz   : müşteri ile kampanyanın BİRLİKTE ürettiği sinyaller

Çapraz blok modelin kalbidir. "Müşteri market harcıyor" tek başına bir şey
söylemez, "kampanya market kampanyası" da öyle; ikisinin kesişimi söyler. Tek
modelin tüm kampanyaları skorlayabilmesinin sebebi de budur.

SIZINTIYA KARŞI İKİ KURAL — ikisi de `tests/test_propensity.py` ile bağlanmıştır:

  1. `responded_at` ve `interaction_id` eğitim çerçevesine HİÇ girmez.
     `responded_at`, kabul eden her müşteride doludur; etiketin kendisidir.
  2. Bölme MÜŞTERİ BAZINDA yapılır (`GroupShuffleSplit`), satır bazında değil.
     Bir müşterinin ~6 teklifinde 70 müşteri kolonu birebir aynıdır; satır
     bazlı bölmede ağaç o vektörü tanıyıp kişinin gizli eğilimini ezberler.

     Sızıntının BÜYÜKLÜĞÜ modelin kapasitesine bağlı ve bu ölçüm projenin en
     öğretici sonucu: `num_leaves=31, lr=0.05` ile rastgele bölme AUC'yi +0.09
     şişirip üretecin gözlemlenebilir sinyal TAVANININ (0.784) üstüne çıkarıyor
     (0.83) — tavanı aşan bir skor öğrenme değil, ezberdir. Izgara taramasının
     seçtiği sığ modelde (`num_leaves=4`) fark sıfıra iniyor, çünkü 4 yapraklı
     bir ağacın tek bir müşteriyi izole edecek yeri yok. Yani "doğru bölme" ile
     "doğru kapasite" aynı sorunun iki ucu. Karşılaştırma her çalıştırmada
     yeniden ölçülür; çıktıdaki "Sızıntı kontrolü" bölümüne bak.

Modele bilinçli olarak GİRMEYENLER:

  * `campaign_id`   — yukarıdaki "tek model" gerekçesi.
  * `offered_at`    — skorlama anında karşılığı yok; modele mevsim ezberletir.
  * `behavior_cluster` (Gün 4) — zaten aynı özelliklerden türetilmiş kayıplı bir
    özet; bilgi katmaz. Segment anlatı ve arayüz tarafında kullanılır, modelde
    değil. Bu iki yetenek kasten ayrı tutuluyor.
  * `opt_in_marketing`, `gender`, `city` — bkz. `features/build.py`.

`generator.py` bu modülden ASLA import edilmez. Üretecin `_campaign_attractiveness`
fonksiyonu hazır durur ve caziptir; onu çağırmak veri üretim kuralını modele elle
kopyalamak olur ve gerçek veriye geçince anlamsızlaşır. Cazibeyi model ödül/bütçe
özniteliklerinden kendisi öğrenir.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import load_config, resolve_path
from src.data.source import build_data_source
from src.features.build import model_feature_columns
from src.features.categories import CATEGORIES

#: Etiketle doğrudan ilişkili oldukları için eğitim çerçevesine hiç alınmayan
#: interaction kolonları. `responded_at` kabul edenlerde her zaman dolu.
LEAK_COLUMNS: set[str] = {"responded_at", "interaction_id"}

#: Interaction tablosundan alınan tek kolonlar.
PAIR_COLUMNS: list[str] = ["customer_id", "campaign_id", "accepted"]

#: Kampanya bloğu — katalog alanlarından doğrudan ya da basit türevle gelir.
CAMPAIGN_NUMERIC: list[str] = [
    "reward_value",
    "max_reward",
    "min_spend",
    "duration_days",
    "budget_ratio",
    "priority",
    "min_tenure_months",
    "requires_digital_active",
]

#: Çapraz blok — müşteri ⊕ kampanya.
CROSS_FEATURES: list[str] = [
    "fit_category_share",
    "fit_monthly_amount",
    "fit_category_trend",
    "min_spend_friction",
    "channel_match",
    "tenure_margin",
    "age_margin",
]

#: Kategorik olarak işaretlenecek kolonlar (LightGBM ve HistGB ikisi de
#: pandas `category` dtype'ını doğrudan anlıyor; one-hot'a gerek yok).
CATEGORICAL_FEATURES: list[str] = [
    "customer_segment",
    "income_band",
    "reward_type",
    "offer_channel",
]

#: `min_spend / aylık kategori harcaması` oranının üst sınırı. Amaç sayısal
#: istikrar (payda sıfıra yaklaşınca oran patlar); ağaç modelleri monoton
#: dönüşüme duyarsız olduğu için sınırın tam değeri sonucu değiştirmez.
FRICTION_CAP = 10.0

#: `channel_match` hesabında dijital sayılan kanallar.
DIGITAL_CHANNELS: frozenset[str] = frozenset({"push", "mobil_app", "email"})

#: Gün 2'de ölçülen gözlemlenebilir sinyal tavanı. Gruplu bölmede bu değerin
#: belirgin üstüne çıkan bir AUC, başarı değil sızıntı işaretidir.
OBSERVABLE_CEILING = 0.784


# --------------------------------------------------------------------------
# Özellik kurulumu — eğitim ve skorlama AYNI fonksiyonları kullanır
# --------------------------------------------------------------------------

def campaign_feature_frame(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Katalogdan kampanya özniteliklerini türetir.

    `target_categories` bir özellik değildir; çapraz blok onu okuyup sayısala
    çevirir ve sonra çerçeveden düşer.
    """
    df = campaigns.copy()
    valid_from = pd.to_datetime(df["valid_from"])
    valid_to = pd.to_datetime(df["valid_to"])

    df["duration_days"] = (valid_to - valid_from).dt.days
    # Kalan bütçe oranı: tükenmiş kampanya (KMP014) 0 alır. Mutlak bütçe yerine
    # oran, farklı büyüklükteki kampanyaları kıyaslanabilir kılıyor.
    df["budget_ratio"] = df["remaining_budget"] / df["total_budget"].replace(0, np.nan)
    df["budget_ratio"] = df["budget_ratio"].fillna(0.0)
    df["requires_digital_active"] = df["requires_digital_active"].astype(int)

    # `min_age` ve `target_categories` özellik değil, çapraz blok için taşınıyor;
    # `add_cross_features` ikisini de tüketip çerçeveden düşürür.
    tutulacak = ["campaign_id", "target_categories", "min_age", "reward_type", "offer_channel"]
    return df[tutulacak + [c for c in CAMPAIGN_NUMERIC if c in df.columns]].copy()


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Müşteri ⊕ kampanya sinyalleri. `df` iki bloğun birleştirilmiş hâli olmalı."""
    hedefte = pd.DataFrame(
        {
            kategori: df["target_categories"].map(lambda lst, k=kategori: k in lst)
            for kategori in CATEGORIES
        },
        index=df.index,
    ).to_numpy(dtype=float)

    def _hedef_toplami(onek: str) -> np.ndarray:
        blok = df[[f"{onek}{k}" for k in CATEGORIES]].to_numpy(dtype=float)
        return (blok * hedefte).sum(axis=1)

    # Kampanyanın hedeflediği kategorilerde müşterinin payı / aylık tutarı /
    # kısa vadeli eğilimi. Payı "ilgi", tutarı "min_spend'i karşılayabilir mi",
    # trendi "harcaması şu an oraya mı kayıyor" sorusuna karşılık gelir.
    df["fit_category_share"] = _hedef_toplami("cat_share_")
    df["fit_monthly_amount"] = _hedef_toplami("cat_monthly_")
    df["fit_category_trend"] = _hedef_toplami("cat_trend_")

    df["min_spend_friction"] = np.minimum(
        df["min_spend"] / np.maximum(df["fit_monthly_amount"], 1.0), FRICTION_CAP
    )

    # Kanal uyumu: dijital müşteriye dijital kanaldan, değilse SMS'ten ulaşmak.
    dijital = df["offer_channel"].isin(DIGITAL_CHANNELS)
    df["channel_match"] = np.where(
        df["digital_active"].astype(bool), dijital, ~dijital
    ).astype(int)

    # Uygunluk eşiklerine ne kadar uzak. Eşiği geçmeyen müşteriye zaten teklif
    # gitmez (`check_eligibility`), ama "sınırda" olmak kabulü etkiliyor.
    df["tenure_margin"] = df["tenure_months"] - df["min_tenure_months"]
    df["age_margin"] = df["age"] - df["min_age"]

    return df.drop(columns=["target_categories", "min_age"])


def build_pair_frame(
    pairs: pd.DataFrame,
    features: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    """(müşteri, kampanya) çiftlerini tam özellik çerçevesine çevirir.

    Eğitimde `pairs` geçmiş tekliflerdir; skorlamada bir müşteri × uygun
    kampanyalar çarpımıdır. İkisinin AYNI fonksiyondan geçmesi bilinçli:
    eğitim ve servis arasında özellik kayması (train/serve skew) böyle önlenir.
    """
    sizinti = LEAK_COLUMNS & set(pairs.columns)
    if sizinti:
        raise ValueError(
            f"Etiket sızıntısı: {sorted(sizinti)} eğitim çerçevesine giremez. "
            f"Interaction tablosundan yalnızca {PAIR_COLUMNS} alınır."
        )

    df = pairs.merge(features, on="customer_id", how="inner")
    eksik = len(pairs) - len(df)
    if eksik:
        raise ValueError(
            f"{eksik} çiftin müşterisi özellik tablosunda yok. "
            f"Önce: python -m src.features.build"
        )

    df = df.merge(campaign_feature_frame(campaigns), on="campaign_id", how="inner")
    return add_cross_features(df)


def feature_columns(df: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    """Modele verilecek kolonlar. Yasak kolonlar burada son kez elenir."""
    kolonlar = (
        model_feature_columns(features) + CAMPAIGN_NUMERIC + CROSS_FEATURES
        + ["reward_type", "offer_channel"]
    )
    kolonlar = [c for c in dict.fromkeys(kolonlar) if c in df.columns]

    yasak = {"campaign_id", "customer_id", "accepted", "offered_at"} | LEAK_COLUMNS
    kacak = yasak & set(kolonlar)
    if kacak:
        raise ValueError(f"Bu kolonlar modele giremez: {sorted(kacak)}")
    return kolonlar


def prepare_matrix(df: pd.DataFrame, kolonlar: list[str]) -> pd.DataFrame:
    """Kategorik kolonları `category`, bool'ları int yapar."""
    X = df[kolonlar].copy()
    for kolon in CATEGORICAL_FEATURES:
        if kolon in X.columns:
            X[kolon] = X[kolon].astype("category")
    for kolon in X.columns:
        if X[kolon].dtype == bool:
            X[kolon] = X[kolon].astype(int)
    return X


def build_training_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Eğitim çerçevesi, X, y ve gruplar (customer_id) döner."""
    source = build_data_source(config)
    features_path = resolve_path(config["data"]["processed_dir"]) / "customer_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} yok. Önce: python -m src.features.build"
        )
    features = pd.read_parquet(features_path)

    interactions = source.get_interactions()
    # Sadece bu üç kolon: responded_at burada bilinçli olarak düşürülüyor.
    pairs = interactions[PAIR_COLUMNS].copy()
    pairs["accepted"] = pairs["accepted"].astype(int)

    df = build_pair_frame(pairs, features, source.get_campaigns())
    kolonlar = feature_columns(df, features)
    return df, prepare_matrix(df, kolonlar), df["accepted"].to_numpy(), df["customer_id"].to_numpy()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def make_model(algorithm: str, cfg: dict):
    """Config'teki algoritmaya göre sınıflandırıcıyı kurar."""
    ortak = dict(
        learning_rate=cfg.get("learning_rate", 0.05),
        random_state=cfg.get("random_state", 42),
    )
    n_estimators = cfg.get("n_estimators", 400)
    num_leaves = cfg.get("num_leaves", 31)
    min_child = cfg.get("min_child_samples", 40)

    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as exc:      # macOS'ta libomp eksikse OSError
            raise RuntimeError(
                "LightGBM yüklenemedi. macOS'ta genelde eksik olan OpenMP kütüphanesidir:\n"
                # conda-forge'da paketin adı llvm-openmp'tir; "libomp" Homebrew adıdır
                # ve conda ile aranırsa PackagesNotFoundError verir.
                "    conda install -n kampanya -c conda-forge llvm-openmp -y\n"
                "veya: brew install libomp\n"
                "Kurmadan devam etmek için: python -m src.models.propensity --algorithm hist\n"
                f"(özgün hata: {exc})"
            ) from exc
        return LGBMClassifier(
            n_estimators=n_estimators, num_leaves=num_leaves,
            min_child_samples=min_child, verbose=-1, **ortak,
        )

    if algorithm == "hist":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=n_estimators, max_leaf_nodes=num_leaves,
            min_samples_leaf=min_child, categorical_features="from_dtype", **ortak,
        )

    raise ValueError(f"Bilinmeyen algoritma: {algorithm!r}. 'lightgbm' veya 'hist' olmalı.")


def split_by_customer(groups: np.ndarray, test_size: float, random_state: int):
    """Müşteri bazlı bölme. Bir müşterinin tüm satırları aynı tarafta kalır."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(gss.split(np.arange(len(groups)), groups=groups))


def tune(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray,
    cfg: dict, algorithm: str,
) -> tuple[dict, list[dict]]:
    """Hiperparametreleri EĞİTİM bölümünün içinden ayrılan doğrulamada seçer.

    Test bölümünde ızgara taraması yapmak, bölme sızıntısının başka kılığıdır:
    raporlanan sayı "en iyi ihtimalle ne olur" olur, "yeni veride ne olur" değil.
    Doğrulama bölmesi de müşteri bazlıdır.

    Neden sığ ağaçlar kazanıyor (varsayılan ızgara buna göre kurulu): 87 özelliğin
    70'i müşteri başına SABİT. Derin bir ağaç bu sabitleri birleştirip tek tek
    müşterileri izole edebiliyor ve eğitimdeki kişilerin kabul oranını ezberliyor;
    bu bilgi yeni müşteriye taşınmadığı için test AUC'si düşüyor. Yaprak sayısını
    kısmak modeli "kişiyi tanıma" yolundan zorla çıkarıyor.
    """
    tr_pos, val_pos = split_by_customer(
        groups[train_idx], cfg.get("valid_size", 0.25), cfg["random_state"]
    )
    tr, val = train_idx[tr_pos], train_idx[val_pos]

    izgara = cfg.get("param_grid") or {}
    kombinasyonlar = [
        {"learning_rate": lr, "num_leaves": nl, "n_estimators": ne}
        for lr in izgara.get("learning_rate", [cfg["learning_rate"]])
        for nl in izgara.get("num_leaves", [cfg["num_leaves"]])
        for ne in izgara.get("n_estimators", [cfg["n_estimators"]])
    ]

    sonuclar: list[dict] = []
    for params in kombinasyonlar:
        model = make_model(algorithm, {**cfg, **params})
        model.fit(X.iloc[tr], y[tr])
        auc = float(roc_auc_score(y[val], model.predict_proba(X.iloc[val])[:, 1]))
        sonuclar.append({**params, "valid_auc": auc})

    sonuclar.sort(key=lambda r: r["valid_auc"], reverse=True)
    en_iyi = {k: v for k, v in sonuclar[0].items() if k != "valid_auc"}
    return en_iyi, sonuclar


# --------------------------------------------------------------------------
# Değerlendirme
# --------------------------------------------------------------------------

def precision_at_1(
    customer_ids: np.ndarray, y: np.ndarray, scores: np.ndarray, min_offers: int
) -> tuple[float, int]:
    """Müşteri başına en yüksek skorlu teklif kabul edilmiş mi?

    Sistem sahada tek tek teklifleri değil, müşteriye gösterilecek SIRALAMAYI
    üretiyor; AUC bunu ölçmez, bu metrik ölçer. Tek teklif almış müşteride
    sıralama diye bir şey olmadığı için `min_offers` altındakiler elenir.
    """
    df = pd.DataFrame({"customer_id": customer_ids, "y": y, "score": scores})
    sayim = df.groupby("customer_id")["y"].transform("size")
    df = df[sayim >= min_offers]
    if df.empty:
        return float("nan"), 0
    en_iyi = df.loc[df.groupby("customer_id")["score"].idxmax()]
    return float(en_iyi["y"].mean()), int(en_iyi.shape[0])


def evaluate(
    customer_ids: np.ndarray, y: np.ndarray, scores: np.ndarray, min_offers: int
) -> dict:
    p1, n_musteri = precision_at_1(customer_ids, y, scores, min_offers)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "precision_at_1": p1,
        "n_ranked_customers": n_musteri,
    }


def baseline_scores(df_train: pd.DataFrame, df_test: pd.DataFrame, random_state: int) -> dict:
    """Modelin kıyaslanacağı üç referans.

    * rastgele    — sıralamanın alt sınırı.
    * popülerlik  — kampanyanın geçmiş kabul oranı. Bankanın bugünkü pratiğine
                    en yakın referans: "en çok tutan kampanyayı herkese ver".
                    Oranlar YALNIZCA eğitim bölümünden hesaplanır.
    * kategori    — "harcadığı kategorinin kampanyasını ver" kuralı; modelsiz,
                    tek satırlık iş kuralı. Model bunu geçemiyorsa gereksizdir.
    """
    rng = np.random.default_rng(random_state)
    oranlar = df_train.groupby("campaign_id")["accepted"].mean()
    genel = float(df_train["accepted"].mean())

    return {
        "rastgele": rng.random(len(df_test)),
        "kampanya_populerligi": df_test["campaign_id"].map(oranlar).fillna(genel).to_numpy(),
        "kategori_kurali": df_test["fit_category_share"].to_numpy(),
    }


# --------------------------------------------------------------------------
# Skorlama (Gün 6+ araç katmanı buradan çağırır)
# --------------------------------------------------------------------------

def load_package(config: dict | None = None) -> dict:
    config = config or load_config()
    path = resolve_path(config["models"]["dir"]) / "propensity.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Önce: python -m src.models.propensity")
    return joblib.load(path)


def score_campaigns(
    package: dict,
    customer_features: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    """Bir müşteri için kampanyaları skorlar; `campaign_id` + `score` döner.

    `customer_features` tek satırlık özellik tablosudur. Uygunluk filtresi
    BURADA YAPILMAZ — `check_eligibility` bu fonksiyondan önce çalışır ve
    `campaigns` zaten elenmiş listedir. Model uygunluğa karar vermez.
    """
    if len(customer_features) != 1:
        raise ValueError("score_campaigns tek müşteri bekler (1 satırlık tablo).")
    if campaigns.empty:
        return pd.DataFrame(columns=["campaign_id", "score"])

    pairs = pd.DataFrame({
        "customer_id": customer_features["customer_id"].iloc[0],
        "campaign_id": campaigns["campaign_id"].to_numpy(),
    })
    df = build_pair_frame(pairs, customer_features, campaigns)
    X = prepare_matrix(df, package["feature_columns"])
    for kolon in CATEGORICAL_FEATURES:
        if kolon in X.columns and kolon in package["categories"]:
            X[kolon] = X[kolon].cat.set_categories(package["categories"][kolon])

    df["score"] = package["model"].predict_proba(X)[:, 1]
    return df[["campaign_id", "score"]].sort_values("score", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------
# Eğitim akışı
# --------------------------------------------------------------------------

def train(
    config: dict, algorithm: str | None = None, do_tune: bool = True
) -> tuple[dict, dict]:
    """Modeli eğitir; (paket, rapor) döner."""
    cfg = config["models"]["propensity"]
    algorithm = algorithm or cfg["algorithm"]
    random_state = cfg["random_state"]
    min_offers = cfg.get("min_offers_for_ranking", 2)

    df, X, y, groups = build_training_data(config)
    tr, te = split_by_customer(groups, cfg["test_size"], random_state)

    if do_tune:
        params, tarama = tune(X, y, groups, tr, cfg, algorithm)
    else:
        params = {k: cfg[k] for k in ("learning_rate", "num_leaves", "n_estimators")}
        tarama = []

    model = make_model(algorithm, {**cfg, **params})
    model.fit(X.iloc[tr], y[tr])
    skor = model.predict_proba(X.iloc[te])[:, 1]

    df_tr, df_te = df.iloc[tr], df.iloc[te]
    metrikler = evaluate(groups[te], y[te], skor, min_offers)

    baselines = {
        ad: evaluate(groups[te], y[te], s, min_offers)
        for ad, s in baseline_scores(df_tr, df_te, random_state).items()
    }
    rastgele_p1 = baselines["rastgele"]["precision_at_1"]
    metrikler["uplift_vs_random"] = (
        metrikler["precision_at_1"] / rastgele_p1 if rastgele_p1 else float("nan")
    )

    # --- sızıntı kontrolü: aynı model, tek fark bölme yöntemi ---
    tr_r, te_r = train_test_split(
        np.arange(len(y)), test_size=cfg["test_size"], random_state=random_state, stratify=y
    )
    model_r = make_model(algorithm, {**cfg, **params})
    model_r.fit(X.iloc[tr_r], y[tr_r])
    auc_rastgele = float(roc_auc_score(y[te_r], model_r.predict_proba(X.iloc[te_r])[:, 1]))
    ortak_musteri = len(set(groups[tr_r]) & set(groups[te_r]))

    paket = {
        "model": model,
        "algorithm": algorithm,
        "params": params,
        "tuning": tarama,
        "feature_columns": list(X.columns),
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in X.columns],
        # Skorlama anında tek müşterinin verisi tüm kategorileri içermez;
        # kategori sıralaması eğitimdekiyle aynı olmazsa model yanlış kod okur.
        "categories": {
            c: list(X[c].cat.categories) for c in CATEGORICAL_FEATURES if c in X.columns
        },
        "metrics": metrikler,
        "baselines": baselines,
        "leakage_check": {
            "grouped_auc": metrikler["auc"],
            "random_split_auc": auc_rastgele,
            "gap": auc_rastgele - metrikler["auc"],
            "shared_customers": ortak_musteri,
            "observable_ceiling": OBSERVABLE_CEILING,
        },
        "n_rows": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "acceptance_rate": float(y.mean()),
        "trained_at": date.today().isoformat(),
    }
    rapor = {"X": X, "y": y, "test_idx": te, "df": df}
    return paket, rapor


def permutation_report(paket: dict, rapor: dict, n_repeats: int = 3, top: int = 15) -> pd.Series:
    """Permütasyon önemi — hangi özellik olmadan AUC düşüyor."""
    from sklearn.inspection import permutation_importance

    te = rapor["test_idx"]
    sonuc = permutation_importance(
        paket["model"], rapor["X"].iloc[te], rapor["y"][te],
        n_repeats=n_repeats, random_state=42, scoring="roc_auc", n_jobs=1,
    )
    return pd.Series(
        sonuc.importances_mean, index=rapor["X"].columns
    ).sort_values(ascending=False).head(top)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kampanya kabul eğilimi modeli.")
    parser.add_argument("--algorithm", choices=["lightgbm", "hist"],
                        help="config.yaml'daki algoritmayı geçersiz kılar")
    parser.add_argument("--importance", action="store_true",
                        help="permütasyon önemlerini hesapla (yavaş)")
    parser.add_argument("--no-tune", action="store_true",
                        help="ızgara taramasını atla, config'teki sabit değerleri kullan")
    args = parser.parse_args(argv)

    config = load_config()
    try:
        paket, rapor = train(config, args.algorithm, do_tune=not args.no_tune)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    m, lk = paket["metrics"], paket["leakage_check"]

    print("\n=== Kampanya Kabul Eğilimi (Propensity) ===\n")
    print(f"  algoritma            : {paket['algorithm']}")
    print(f"  eğitim çifti         : {paket['n_rows']:,} "
          f"({paket['n_customers']:,} müşteri, kabul %{paket['acceptance_rate']*100:.1f})")
    print(f"  özellik              : {len(paket['feature_columns'])}")
    print(f"  hiperparametre       : " + ", ".join(f"{k}={v}" for k, v in paket["params"].items()))
    if paket["tuning"]:
        en_kotu = paket["tuning"][-1]
        print(f"  ızgara               : {len(paket['tuning'])} kombinasyon, "
              f"doğrulama AUC {en_kotu['valid_auc']:.4f} - "
              f"{paket['tuning'][0]['valid_auc']:.4f}")

    print("\n--- Model (müşteri-gruplu bölme) ---")
    print(f"  AUC                  : {m['auc']:.4f}")
    print(f"  PR-AUC               : {m['pr_auc']:.4f}   (taban {paket['acceptance_rate']:.3f})")
    print(f"  precision@1          : {m['precision_at_1']:.4f}   "
          f"({m['n_ranked_customers']} müşteri üzerinde)")
    print(f"  rastgeleye karşı     : {m['uplift_vs_random']:.2f}x")

    print("\n--- Baseline karşılaştırması ---")
    print(f"  {'yöntem':<24}{'AUC':>8}{'PR-AUC':>9}{'p@1':>8}")
    for ad, b in paket["baselines"].items():
        print(f"  {ad:<24}{b['auc']:>8.4f}{b['pr_auc']:>9.4f}{b['precision_at_1']:>8.4f}")
    print(f"  {'MODEL':<24}{m['auc']:>8.4f}{m['pr_auc']:>9.4f}{m['precision_at_1']:>8.4f}")

    print("\n--- Sızıntı kontrolü (aynı model, tek fark bölme yöntemi) ---")
    print(f"  rastgele satır bölmesi : AUC {lk['random_split_auc']:.4f}   "
          f"(hem eğitimde hem testte olan müşteri: {lk['shared_customers']:,})")
    print(f"  müşteri-gruplu bölme   : AUC {lk['grouped_auc']:.4f}   (raporlanan)")
    print(f"  fark                   : {lk['gap']:+.4f}")
    print(f"  gözlemlenebilir tavan  : {lk['observable_ceiling']:.3f}")
    if lk["random_split_auc"] > OBSERVABLE_CEILING:
        print("  -> Rastgele bölme tavanı AŞIYOR: bu fazlalık öğrenme değil, "
              "müşteri ezberidir.")
    if lk["grouped_auc"] > OBSERVABLE_CEILING + 0.02:
        print("  [!] Gruplu bölme de tavanın üstünde — özellik setinde sızıntı ara.")

    if args.importance:
        print("\n--- Permütasyon önemi (AUC düşüşü) ---")
        for ad, deger in permutation_report(paket, rapor).items():
            print(f"  {ad:<28}{deger:>8.4f}")

    model_dir = resolve_path(config["models"]["dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "propensity.joblib"
    joblib.dump(paket, path)
    print(f"\n  yazıldı: {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
