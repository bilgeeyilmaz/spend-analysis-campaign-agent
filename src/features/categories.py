"""Kategori, kanal ve taksit davranışı özellikleri.

Projenin öneri mantığının kalbi burası: "bu müşteri parasını nereye harcıyor"
sorusunun sayısal cevabı. Kategori payları hem propensity modelinin en güçlü
girdisi hem de agent'ın Türkçe gerekçesinin kaynağıdır ("harcamanızın %38'i
markette, o yüzden market kampanyası").

Tasarım notu — neden pay, neden tutar değil: iki müşteri aynı tutarı markete
harcayabilir ama biri için bu bütçesinin %60'ı, diğeri için %5'idir. Kampanya
ilgisini belirleyen orandır. Mutlak tutar da ayrıca tutuluyor, çünkü kampanyanın
`min_spend` koşulunu karşılayıp karşılamayacağını o belirler.
"""

from __future__ import annotations

import pandas as pd

from src.data.schemas import Channel, MccCategory

EPS = 1e-9

#: Kolon adları şemadaki enum'dan türetilir — kategori eklendiğinde özellik
#: tablosu kendiliğinden genişler, burada elle liste tutulmaz.
CATEGORIES: list[str] = [c.value for c in MccCategory]
CHANNELS: list[str] = [c.value for c in Channel]


def _pivot_share(
    df: pd.DataFrame, sutun: str, kategoriler: list[str], onek: str, degerler: str = "amount"
) -> pd.DataFrame:
    """Müşteri × kategori tutar tablosu -> pay tablosu (satır toplamı 1)."""
    pivot = df.pivot_table(
        index="customer_id", columns=sutun, values=degerler, aggfunc="sum", observed=True
    )
    # Veride hiç görülmemiş kategori de kolon olarak dursun: özellik tablosunun
    # şekli veri örneklemine göre değişmemeli, yoksa model dosyası uyumsuz kalır.
    pivot = pivot.reindex(columns=kategoriler, fill_value=0.0).fillna(0.0)
    paylar = pivot.div(pivot.sum(axis=1) + EPS, axis=0)
    paylar.columns = [f"{onek}{c}" for c in paylar.columns]
    return paylar


def category_features(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    recent_days: int,
    long_days: int,
) -> pd.DataFrame:
    """Kategori payları, mutlak tutarlar ve kısa/uzun vade kategori trendi."""
    paylar = _pivot_share(transactions, "mcc_category", CATEGORIES, "cat_share_")

    # Aylık ortalama kategori tutarı: kampanyanın min_spend koşuluyla doğrudan
    # kıyaslanabilen tek metrik budur.
    n_months = max(
        (as_of.to_period("M") - transactions["transaction_date"].min().to_period("M")).n, 1
    )
    tutarlar = transactions.pivot_table(
        index="customer_id", columns="mcc_category", values="amount",
        aggfunc="sum", observed=True,
    ).reindex(columns=CATEGORIES, fill_value=0.0).fillna(0.0)
    aylik = tutarlar / n_months
    aylik.columns = [f"cat_monthly_{c}" for c in aylik.columns]

    # Trend: son 30 günün kategori payı, son 90 güne göre nereye kayıyor.
    # Pay farkı kullanıyoruz (tutar oranı değil): tutar oranı, toplam harcaması
    # artan müşteride her kategoriyi birden "yükseliyor" gösterir.
    son = transactions[transactions["transaction_date"] >= as_of - pd.Timedelta(days=recent_days)]
    uzun = transactions[transactions["transaction_date"] >= as_of - pd.Timedelta(days=long_days)]
    pay_son = _pivot_share(son, "mcc_category", CATEGORIES, "t_")
    pay_uzun = _pivot_share(uzun, "mcc_category", CATEGORIES, "t_")
    trend = pay_son.reindex(pay_uzun.index.union(pay_son.index)).fillna(0.0).sub(
        pay_uzun.reindex(pay_uzun.index.union(pay_son.index)).fillna(0.0)
    )
    trend.columns = [c.replace("t_", "cat_trend_") for c in trend.columns]

    out = paylar.join(aylik, how="outer").join(trend, how="outer").fillna(0.0)

    # Yorumlanabilirlik ve agent metni için: en büyük kategori ve payı.
    # Model girdisi DEĞİL (bkz. build.EXCLUDED_FEATURES) — string alan.
    pay_kolonlari = [f"cat_share_{c}" for c in CATEGORIES]
    out["top_category"] = out[pay_kolonlari].idxmax(axis=1).str.replace("cat_share_", "", regex=False)
    out["top_category_share"] = out[pay_kolonlari].max(axis=1)

    return out.reset_index()


def channel_and_installment_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Kanal karması ve taksit kullanım profili.

    Kanal karması dijital olgunluğun davranışsal ölçüsüdür: `digital_active`
    bayrağı "giriş yaptı mı" der, online/mobil payı "gerçekten dijital mi
    yaşıyor" der. Taksit kullanımı ise taksit tipli kampanyalara (KMP004, KMP006,
    KMP008) yatkınlığın en doğrudan göstergesidir.
    """
    kanal = _pivot_share(transactions, "channel", CHANNELS, "channel_share_")

    grouped = transactions.groupby("customer_id", observed=True)
    taksit = pd.DataFrame({
        "installment_ratio": grouped["installment_count"].apply(lambda s: (s > 1).mean()),
        "avg_installment": grouped["installment_count"].apply(lambda s: s[s > 1].mean()),
        "max_installment": grouped["installment_count"].max(),
    })
    # Hiç taksit kullanmayan müşteride ortalama tanımsız -> 1 (tek çekim).
    taksit["avg_installment"] = taksit["avg_installment"].fillna(1.0)

    return kanal.join(taksit, how="outer").fillna(0.0).reset_index()
