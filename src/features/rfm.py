"""RFM ve genel harcama metrikleri.

RFM (Recency / Frequency / Monetary) perakende ve bankacılıkta müşteri değerini
özetlemenin standart yoludur. Burada klasik RFM'e iki ek yapıyoruz:

  * **Pencereli metrikler** (30 / 90 gün) — "ne kadar harcadı" kadar "SON ZAMANDA
    ne kadar harcadı" da önemli; kampanya kararı güncel davranışa bakar.
  * **Momentum** — son 30 günün, 90 günlük ortalamaya oranı. Harcaması hızlanan
    müşteri ile yavaşlayan müşteri aynı toplam tutara sahip olabilir.

Bu modül `transactions` dışında hiçbir şey bilmez: müşteri tablosuna, kampanyaya
ya da modele bağımlı değildir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Sıfıra bölmeyi engellemek için kullanılan küçük sabit.
EPS = 1e-9


def compute_rfm(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    recent_days: int,
    long_days: int,
) -> pd.DataFrame:
    """Müşteri başına RFM + hacim + momentum metrikleri.

    `as_of` referans tarihidir: recency ondan geriye sayılır. Gerçek veriye
    geçildiğinde "bugün" yerine veri kesim tarihi verilir, böylece geçmişe dönük
    yeniden hesaplama (backfill) aynı sonucu üretir.
    """
    df = transactions
    recent_start = as_of - pd.Timedelta(days=recent_days)
    long_start = as_of - pd.Timedelta(days=long_days)

    grouped = df.groupby("customer_id", observed=True)
    out = grouped.agg(
        recency_days=("transaction_date", lambda s: (as_of - s.max()).days),
        frequency_total=("amount", "size"),
        monetary_total=("amount", "sum"),
        avg_ticket=("amount", "mean"),
        max_ticket=("amount", "max"),
        std_ticket=("amount", "std"),
        first_transaction=("transaction_date", "min"),
        distinct_categories=("mcc_category", "nunique"),
        distinct_merchants=("merchant_id", "nunique"),
    )
    out["std_ticket"] = out["std_ticket"].fillna(0.0)

    # Aktif ay sayısı: kaç farklı takvim ayında işlem var. "12 ayda 100 işlem"
    # ile "1 ayda 100 işlem" çok farklı müşterilerdir.
    aylar = df.assign(ay=df["transaction_date"].dt.to_period("M"))
    out["active_months"] = aylar.groupby("customer_id", observed=True)["ay"].nunique()

    # Pencereli metrikler
    for etiket, baslangic in (("30d", recent_start), ("90d", long_start)):
        pencere = df[df["transaction_date"] >= baslangic]
        agg = pencere.groupby("customer_id", observed=True)["amount"].agg(["size", "sum"])
        out[f"frequency_{etiket}"] = agg["size"]
        out[f"monetary_{etiket}"] = agg["sum"]

    out = out.fillna({
        "frequency_30d": 0, "monetary_30d": 0.0,
        "frequency_90d": 0, "monetary_90d": 0.0,
    })

    # Momentum: son 30 gün, 90 günlük aylık ortalamaya göre nerede.
    # 1.0 = sabit, >1 = hızlanıyor. Kampanya zamanlaması için en okunaklı sinyal.
    aylik_ortalama_90 = out["monetary_90d"] * (recent_days / long_days)
    out["spend_momentum"] = out["monetary_30d"] / (aylik_ortalama_90 + EPS)
    out["spend_momentum"] = out["spend_momentum"].clip(upper=10.0)

    # Müşterinin bankayla işlem geçmişinin uzunluğu (veri penceresi içinde).
    out["transaction_history_days"] = (as_of - out["first_transaction"]).dt.days
    out = out.drop(columns=["first_transaction"])

    # Log dönüşümü: harcama dağılımı uzun kuyruklu; KMeans gibi uzaklık tabanlı
    # yöntemlerde ham tutar birkaç uç müşteriyi tüm kümelenmeye hâkim kılar.
    out["log_monetary_total"] = np.log1p(out["monetary_total"])
    out["log_frequency_total"] = np.log1p(out["frequency_total"])

    return out.reset_index()
