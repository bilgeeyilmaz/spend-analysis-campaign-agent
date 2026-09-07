"""Sentetik veri üreteci.

Üretilen dosyalar (`data/raw/`): `customers.json`, `transactions.json`,
`interactions.json`. **`campaigns.json` üretilmez** — katalog elle yazıldı ve
versiyonlanıyor; bu üreteç onu sadece okur.

Tasarımın iki taşıyıcı fikri var:

1. **Gizli persona.** Her müşterinin harcama profili, veriye HİÇ yazılmayan bir
   persona'dan türer (dijital genç, aile-market odaklı, seyahat eden affluent...).
   Gün 4'te KMeans'in bulması beklenen yapı budur; "model gizli yapıyı geri buldu"
   cümlesi ancak yapı gerçekten gizliyse anlamlıdır.

2. **Etiket bir kural değildir.** `Interaction.accepted`, gözlemlenebilir sinyaller
   (kategori uyumu, kampanya cazibesi, kanal uyumu, asgari harcama sürtünmesi) ile
   birlikte **gizli bir müşteri eğilimi** ve gürültüden üretilir. Gizli eğilim özellik
   tablosuna girmediği için modelin ulaşabileceği bir tavan vardır — hedef bant
   AUC 0.72–0.82. Bu tavan çalışma sonunda ölçülüp rapor edilir.

Ayrıca `analyze_spending` aracının (Gün 6) bulacağı iki yapı bilinçli olarak ekilir:
düzenli (recurring) ödemeler ve son 3 ay içindeki olağandışı kategori sıçramaları.
Bunlar üretilmezse araç boş çıktı verir ve tüm veriyi yeniden üretmek gerekir.

Çalıştırma:
    python -m src.data.generator                      # config.yaml'daki ayarlarla
    python -m src.data.generator --n-customers 200    # hızlı duman testi
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from src.config import load_config, resolve_path
from src.data.schemas import (
    Campaign,
    Channel,
    Customer,
    CustomerSegment,
    IncomeBand,
    Interaction,
    MccCategory,
    OfferChannel,
    Transaction,
)

# --------------------------------------------------------------------------
# Sabitler — gerçekçilik parametreleri
# --------------------------------------------------------------------------

#: Nüfusa kabaca orantılı şehir dağılımı. Şehir bir hedefleme özelliği DEĞİL,
#: sadece veri gerçekçiliği için tutuluyor.
CITIES: dict[str, float] = {
    "İstanbul": 0.24, "Ankara": 0.09, "İzmir": 0.07, "Bursa": 0.05,
    "Antalya": 0.04, "Konya": 0.03, "Adana": 0.03, "Gaziantep": 0.03,
    "Şanlıurfa": 0.02, "Kocaeli": 0.03, "Mersin": 0.03, "Kayseri": 0.02,
    "Samsun": 0.02, "Trabzon": 0.02, "Eskişehir": 0.02, "Diyarbakır": 0.02,
    "Denizli": 0.02, "Malatya": 0.01, "Erzurum": 0.01, "Van": 0.01,
    "Sakarya": 0.02, "Manisa": 0.02, "Balıkesir": 0.02, "Aydın": 0.02,
    "Tekirdağ": 0.02, "Muğla": 0.02, "Hatay": 0.02, "Sivas": 0.01,
    "Zonguldak": 0.01, "Çanakkale": 0.01, "Ordu": 0.01, "Elazığ": 0.01,
}

#: Kategori -> gerçek hayatta o kategoriye düşen MCC kodları.
#: Ham kod şemada saklanıyor (`Transaction.mcc_code`); model kategoriyi kullanır.
MCC_CODES: dict[str, list[str]] = {
    "market": ["5411", "5422", "5451", "5499"],
    "akaryakit": ["5541", "5542"],
    "restoran": ["5812", "5813", "5814"],
    "giyim": ["5651", "5621", "5661", "5691"],
    "elektronik": ["5732", "5722", "5734"],
    "seyahat": ["4511", "4722", "7011", "4111"],
    "saglik": ["8011", "8021", "8062", "5912"],
    "egitim": ["8220", "8211", "8299"],
    "eglence": ["7832", "7996", "7997", "5815"],
    "telekom": ["4814", "4816", "4899"],
    "online_alisveris": ["5964", "5969", "5942"],
    "diger": ["5999", "7299", "7230"],
}

#: Kategori -> (log-ortalama, log-std). Tutarlar lognormal: uzun kuyruk gerçekçidir
#: ve olağandışı harcama tespitinin anlamlı olması için şart (her işlem ortalamaya
#: yapışıksa "sıçrama" diye bir şey olmaz).
CATEGORY_TICKET: dict[str, tuple[float, float]] = {
    "market": (6.55, 0.60),            # medyan ~700 TL
    "akaryakit": (7.31, 0.45),         # ~1.500 TL
    "restoran": (6.40, 0.70),          # ~600 TL
    "giyim": (7.09, 0.80),             # ~1.200 TL
    "elektronik": (8.29, 0.90),        # ~4.000 TL
    "seyahat": (8.52, 0.90),           # ~5.000 TL
    "saglik": (6.80, 0.80),            # ~900 TL
    "egitim": (7.82, 0.70),            # ~2.500 TL
    "eglence": (6.11, 0.70),           # ~450 TL
    "telekom": (6.21, 0.35),           # ~500 TL
    "online_alisveris": (6.68, 0.85),  # ~800 TL
    "diger": (6.40, 0.80),             # ~600 TL
}

#: Gelir bandına göre sepet büyüklüğü çarpanı (A = en yüksek).
INCOME_MULTIPLIER: dict[str, float] = {"A": 1.9, "B": 1.4, "C": 1.0, "D": 0.7}

#: Ay (1-12) bazlı mevsimsel çarpanlar. Kampanya kataloğuyla uyumlu olmak zorunda:
#: KMP007 "sezon sonu giyim" (5 Ağu – 20 Eyl) ancak veride o dönem giyim harcaması
#: varsa anlamlı bir öneri olur.
SEASONALITY: dict[str, dict[int, float]] = {
    "giyim": {1: 1.35, 7: 1.25, 8: 1.50, 9: 1.40},
    "seyahat": {6: 1.60, 7: 1.90, 8: 1.80, 9: 1.20},
    "egitim": {2: 1.50, 9: 2.20, 10: 1.40},
    "elektronik": {11: 1.90, 12: 1.30},
    "eglence": {7: 1.30, 8: 1.30, 12: 1.20},
    "market": {12: 1.15},
    "akaryakit": {7: 1.20, 8: 1.20},
}

#: Gizli persona'lar. `weight` = nüfustaki pay, `affinity` = config'teki temel
#: kategori ağırlıkları üzerine çarpan, `segment` = ticari segment olasılıkları.
#: DİKKAT: persona adı hiçbir çıktı dosyasına yazılmaz.
PERSONAS: dict[str, dict] = {
    "dijital_genc": {
        "weight": 0.22,
        "affinity": {
            "online_alisveris": 2.5, "eglence": 2.2, "restoran": 1.8, "giyim": 1.4,
            "elektronik": 1.3, "telekom": 1.3, "seyahat": 0.9, "diger": 1.0,
            "market": 0.6, "saglik": 0.5, "egitim": 0.6, "akaryakit": 0.4,
        },
        "segment": {"mass": 0.80, "affluent": 0.19, "private": 0.01},
        "age": (27, 5, 18, 40),
        "digital_active": 0.97,
        "volume": 1.10,
        "amount": 0.85,
        "installment": 1.2,
    },
    "aile_market": {
        "weight": 0.28,
        "affinity": {
            "market": 2.0, "egitim": 1.8, "saglik": 1.6, "giyim": 1.2,
            "akaryakit": 1.1, "telekom": 1.1, "diger": 1.0, "online_alisveris": 0.8,
            "elektronik": 0.8, "restoran": 0.7, "eglence": 0.6, "seyahat": 0.5,
        },
        "segment": {"mass": 0.78, "affluent": 0.20, "private": 0.02},
        "age": (41, 8, 28, 60),
        "digital_active": 0.75,
        "volume": 1.05,
        "amount": 1.00,
        "installment": 1.45,
    },
    "seyahat_affluent": {
        "weight": 0.14,
        "affinity": {
            "seyahat": 3.0, "restoran": 1.8, "elektronik": 1.6, "giyim": 1.6,
            "eglence": 1.4, "online_alisveris": 1.2, "akaryakit": 1.1, "diger": 1.0,
            "saglik": 0.9, "telekom": 0.8, "egitim": 0.7, "market": 0.7,
        },
        "segment": {"mass": 0.25, "affluent": 0.55, "private": 0.20},
        "age": (45, 9, 30, 65),
        "digital_active": 0.90,
        "volume": 1.00,
        "amount": 1.35,
        "installment": 1.0,
    },
    "arac_sahibi_calisan": {
        "weight": 0.20,
        "affinity": {
            "akaryakit": 2.6, "market": 1.3, "diger": 1.3, "restoran": 1.2,
            "telekom": 1.1, "saglik": 0.9, "elektronik": 0.8, "giyim": 0.8,
            "egitim": 0.8, "seyahat": 0.7, "eglence": 0.7, "online_alisveris": 0.7,
        },
        "segment": {"mass": 0.75, "affluent": 0.23, "private": 0.02},
        "age": (38, 9, 24, 58),
        "digital_active": 0.78,
        "volume": 1.00,
        "amount": 1.00,
        "installment": 0.95,
    },
    "temkinli_tasarrufcu": {
        "weight": 0.16,
        "affinity": {
            "market": 1.6, "saglik": 1.3, "telekom": 1.2, "akaryakit": 1.0,
            "diger": 0.7, "giyim": 0.6, "restoran": 0.5, "egitim": 0.5,
            "elektronik": 0.5, "online_alisveris": 0.5, "eglence": 0.4, "seyahat": 0.4,
        },
        "segment": {"mass": 0.80, "affluent": 0.17, "private": 0.03},
        "age": (55, 12, 35, 80),
        "digital_active": 0.45,
        "volume": 0.75,
        "amount": 0.85,
        "installment": 0.35,
    },
}

#: Düzenli ödeme üretilebilecek kategoriler -> (asgari, azami) aylık tutar.
#: Bunlar `analyze_spending.duzenli_giderler` alanının kaynağıdır.
RECURRING_CATEGORIES: dict[str, tuple[float, float]] = {
    "telekom": (300.0, 800.0),      # fatura talimatı -> KMP010'un gerekçesi
    "eglence": (150.0, 400.0),      # dijital abonelik
    "egitim": (1500.0, 4000.0),     # kurs taksiti
    "saglik": (400.0, 1200.0),      # özel sigorta primi
    "diger": (500.0, 1500.0),       # spor salonu vb.
}

#: Düzenli ödemesi olan müşteri oranı. 1.0 YAPMA: düzenli gideri olmayan müşteri,
#: Gün 6'da `analyze_spending` testinin negatif fixture'ıdır (boş liste dönmeli).
RECURRING_RATE = 0.78

#: Olağandışı sıçrama üretilebilecek kategoriler (tek seferlik büyük alım).
SPIKE_CATEGORIES: list[str] = ["elektronik", "seyahat", "giyim", "egitim", "saglik"]

#: Sıçrama yaşayan müşteri oranı.
SPIKE_RATE = 0.20

#: Kategoriye göre kanal dağılımı. Online alışveriş POS'tan geçmez, akaryakıt
#: mobilden ödenmez — kanal karması özellik olarak kullanılacağı için tutarlı olmalı.
CHANNEL_WEIGHTS: dict[str, dict[str, float]] = {
    "online_alisveris": {"online": 0.90, "mobil": 0.10},
    "telekom": {"mobil": 0.55, "online": 0.35, "pos": 0.10},
    "seyahat": {"online": 0.60, "pos": 0.30, "mobil": 0.10},
    "eglence": {"online": 0.55, "pos": 0.35, "mobil": 0.10},
    "egitim": {"online": 0.40, "pos": 0.45, "mobil": 0.15},
    "elektronik": {"pos": 0.55, "online": 0.40, "mobil": 0.05},
    "giyim": {"pos": 0.60, "online": 0.38, "mobil": 0.02},
    "akaryakit": {"pos": 0.97, "mobil": 0.03},
    "market": {"pos": 0.90, "online": 0.09, "mobil": 0.01},
    "restoran": {"pos": 0.85, "online": 0.13, "mobil": 0.02},
    "saglik": {"pos": 0.75, "online": 0.20, "mobil": 0.05},
    "diger": {"pos": 0.70, "online": 0.22, "mobil": 0.08},
}

#: Taksitin yaygın olduğu kategoriler ve olası taksit sayıları.
#: Market ve akaryakıt bilinçli olarak YOK: Türkiye'de bu kategorilerde kredi
#: kartına taksit yasal olarak yapılamaz.
INSTALLMENT_CATEGORIES: dict[str, list[int]] = {
    "elektronik": [3, 6, 9, 12],
    "seyahat": [3, 6, 9],
    "egitim": [6, 9, 12],
    "giyim": [3, 6],
    "saglik": [3, 6],
    "online_alisveris": [3, 6],
    "diger": [3, 6, 9],          # mobilya, beyaz eşya
}

#: Taksitin devreye girdiği asgari tutar.
INSTALLMENT_MIN_AMOUNT = 1000.0

#: Kategori başına üye işyeri havuzu büyüklüğü.
MERCHANTS_PER_CATEGORY = 20

# --------------------------------------------------------------------------
# Etiket üretimi katsayıları — "döngüsellik tuzağı"na karşı kalibrasyon
# --------------------------------------------------------------------------
#
# logit = INTERCEPT
#       + W_FIT      * kategori_uyumu(z)      <- ÖZELLİKLERDEN TÜRETİLEBİLİR
#       + W_ATTRACT  * kampanya_cazibesi(z)   <- ÖZELLİKLERDEN TÜRETİLEBİLİR
#       + W_CHANNEL  * kanal_uyumu(±1)        <- ÖZELLİKLERDEN TÜRETİLEBİLİR
#       - W_FRICTION * asgari_harcama_sürtünmesi(z)
#       + W_LATENT   * gizli_eğilim           <- ÖZELLİK TABLOSUNDA YOK
#       + gürültü
#
# Gizli terim modelin ulaşamayacağı bir tavan yaratır. Katsayılar 900 müşterilik bir
# taramayla ampirik olarak ayarlandı — teorik formül, `fit` ile `friction` arasındaki
# korelasyon yüzünden tavanı olduğundan düşük tahmin ediyor. Ölçülen tavan:
#
#   W_LATENT   1.3    1.8    2.0    2.3    2.8    3.3
#   AUC       0.820  0.792  ~0.78  0.753  0.719  0.694
#
# 2.0 seçildi: 0.72–0.82 bandının ortası, iki kenardan da eşit uzak. Katsayıları
# değiştirirsen çalışma sonundaki "gözlemlenebilir sinyal tavanı" satırına bak.
W_FIT = 1.55
W_ATTRACT = 0.66
W_CHANNEL = 0.25
W_FRICTION = 0.73
W_LATENT = 2.00
#: `config.generator.label_noise` bu katsayıyla çarpılıp gürültünün std'si olur.
NOISE_SCALE = 3.0
#: Kabul oranını makul banda (%20-30) çeken sabit terim.
INTERCEPT = -2.30


# --------------------------------------------------------------------------
# Küçük yardımcılar
# --------------------------------------------------------------------------


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Ağırlıklı sözlükten bir anahtar seçer."""
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standartlaştırma; sabit dizide sıfır döner (sıfıra bölme yok)."""
    std = values.std()
    return np.zeros_like(values) if std < 1e-9 else (values - values.mean()) / std


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Sıralama tabanlı ROC-AUC (sklearn bağımlılığı olmadan).

    Veri katmanına ağır bir bağımlılık eklememek için elle yazıldı; Gün 5'te
    modelin gerçek metriği sklearn ile ölçülecek.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _month_windows(as_of: date, n_months: int) -> list[tuple[date, date]]:
    """Son `n_months` takvim ayının (başlangıç, bitiş) aralıkları — eskiden yeniye.

    Son ay kısmidir: `as_of` gününde biter. Aylık seri (`aylik_seri`) ve düzenli
    ödeme tespiti takvim ayına dayandığı için gün sayısıyla değil ayla çalışıyoruz.
    """
    windows: list[tuple[date, date]] = []
    year, month = as_of.year, as_of.month
    for _ in range(n_months):
        first = date(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        last = min(date(year, month, last_day), as_of)
        windows.append((first, last))
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return list(reversed(windows))


def _random_datetime(rng: np.random.Generator, start: date, end: date) -> datetime:
    """Aralıkta rastgele bir an. Saatler alışveriş saatlerine yığılır."""
    span = (end - start).days
    day = start + timedelta(days=int(rng.integers(0, max(span, 1) + 1)))
    hour = int(rng.choice(
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        p=[0.04, 0.05, 0.06, 0.09, 0.09, 0.07, 0.07, 0.08, 0.09, 0.11, 0.10, 0.07, 0.05, 0.03],
    ))
    return datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)))


# --------------------------------------------------------------------------
# 1) Müşteriler
# --------------------------------------------------------------------------


def generate_customers(
    rng: np.random.Generator,
    n_customers: int,
    as_of: date,
    opt_in_rate: float,
) -> tuple[list[dict], dict[str, dict]]:
    """Müşteri kayıtlarını ve her müşterinin GİZLİ değişkenlerini üretir.

    İkinci dönüş değeri (`latent`) diske YAZILMAZ: persona, gizli kabul eğilimi ve
    aktivite çarpanı burada kalır. Modelin bunlara erişememesi, ölçtüğümüz AUC'nin
    gerçek bir tahmin başarısı olmasını sağlar.
    """
    customers: list[dict] = []
    latent: dict[str, dict] = {}

    persona_weights = {name: cfg["weight"] for name, cfg in PERSONAS.items()}

    for i in range(n_customers):
        customer_id = f"C{i:06d}"
        persona_name = _pick(rng, persona_weights)
        persona = PERSONAS[persona_name]

        segment = _pick(rng, persona["segment"])

        mean_age, sd_age, min_age, max_age = persona["age"]
        age = int(np.clip(rng.normal(mean_age, sd_age), min_age, max_age))

        # Gelir bandı segmentle ilişkili ama birebir değil.
        income_band = _pick(rng, {
            "mass": {"A": 0.02, "B": 0.15, "C": 0.48, "D": 0.35},
            "affluent": {"A": 0.18, "B": 0.52, "C": 0.28, "D": 0.02},
            "private": {"A": 0.70, "B": 0.28, "C": 0.02, "D": 0.00},
        }[segment])

        # Kıdem yaşı aşamaz: en fazla 18 yaşından beri müşteri olabilir.
        max_tenure_years = min(age - 18, 25)
        tenure_months = int(rng.beta(1.6, 2.2) * max_tenure_years * 12)
        created_at = as_of - timedelta(days=int(tenure_months * 30.44))

        digital_active = bool(rng.random() < persona["digital_active"] * (1.15 if age < 35 else 1.0))

        has_credit_card = bool(rng.random() < {"mass": 0.80, "affluent": 0.93, "private": 0.98}[segment])
        has_deposit = bool(rng.random() < {"mass": 0.30, "affluent": 0.55, "private": 0.85}[segment])
        if persona_name == "temkinli_tasarrufcu":
            has_deposit = bool(rng.random() < 0.65)

        customers.append({
            "customer_id": customer_id,
            "age": age,
            "gender": str(rng.choice(["K", "E", "B"], p=[0.48, 0.48, 0.04])),
            "city": _pick(rng, CITIES),
            "customer_segment": segment,
            "income_band": income_band,
            "tenure_months": tenure_months,
            "has_credit_card": has_credit_card,
            "has_debit_card": bool(rng.random() < 0.97),
            "has_loan": bool(rng.random() < (0.35 if persona_name == "aile_market" else 0.22)),
            "has_deposit": has_deposit,
            "digital_active": digital_active,
            "opt_in_marketing": bool(rng.random() < opt_in_rate),
            "created_at": created_at.isoformat(),
        })

        latent[customer_id] = {
            "persona": persona_name,
            # Gizli kabul eğilimi: kampanyadan bağımsız, kişiye özgü "teklife
            # açıklık". Özellik tablosunda karşılığı YOKTUR.
            "propensity": float(rng.normal(0.0, 1.0)),
            # İşlem hacmi heterojenliği (segment içi fark).
            "activity": float(rng.lognormal(0.0, 0.35) * persona["volume"]),
            # Taksit kullanma eğilimi. Müşteriye özgü olması şart: sabit bir
            # olasılıkla üretilirse `installment_ratio` herkeste aynı çıkar,
            # KMeans için bilgisiz bir kolon olur ve taksit tipli kampanyalara
            # (KMP004/006/008) yatkınlık modellenemez.
            "installment_affinity": float(np.clip(
                rng.beta(2.0, 3.0) * persona["installment"], 0.0, 1.0
            )),
            "amount_factor": persona["amount"] * INCOME_MULTIPLIER[income_band],
        }

    return customers, latent


# --------------------------------------------------------------------------
# 2) İşlemler
# --------------------------------------------------------------------------


#: Kategori başına beklenen sepet tutarı: lognormal ortalaması exp(mu + sigma²/2).
MEAN_TICKET: dict[str, float] = {
    cat: float(np.exp(mu + sigma**2 / 2)) for cat, (mu, sigma) in CATEGORY_TICKET.items()
}


def _category_weights_for(
    persona_name: str, base_weights: dict[str, float]
) -> dict[str, float]:
    """Kategori seçim olasılıkları (İŞLEM ADEDİ bazında).

    `config.generator.category_weights` TCMB serisinden geldiği için **tutar payıdır**,
    adet payı değil. Doğrudan seçim olasılığı olarak kullanılırsa seyahat/elektronik
    gibi yüksek sepetli kategoriler tutar payını katlar (ilk denemede seyahat %5 hedefe
    karşı %26 çıkmıştı). Bu yüzden adet olasılığı `pay / ortalama_sepet` ile türetilir:
    o zaman gerçekleşen tutar payı ≈ hedeflenen pay olur.
    """
    affinity = PERSONAS[persona_name]["affinity"]
    counts = {
        cat: base_weights.get(cat, 0.0) * affinity.get(cat, 1.0) / MEAN_TICKET[cat]
        for cat in base_weights
    }
    total = sum(counts.values())
    return {cat: c / total for cat, c in counts.items()}


def _make_recurring(
    rng: np.random.Generator,
    customer_id: str,
    windows: list[tuple[date, date]],
) -> list[dict]:
    """Müşteriye 1-3 düzenli aylık ödeme ekler.

    `analyze_spending.duzenli_giderler` tam olarak bunu bulacak: aynı `merchant_id`,
    ayda bir, sabit güne yakın (±3), tutar oynaması ≤ %5. Tespit eşiği %15 tolerans
    olduğu için üretimdeki %5'lik oynama rahatça yakalanır.
    """
    rows: list[dict] = []
    n_recurring = int(rng.integers(1, 4))
    categories = list(rng.choice(
        list(RECURRING_CATEGORIES), size=min(n_recurring, len(RECURRING_CATEGORIES)),
        replace=False,
    ))

    for slot, category in enumerate(categories):
        low, high = RECURRING_CATEGORIES[category]
        base_amount = float(rng.uniform(low, high))
        base_day = int(rng.integers(1, 29))
        merchant_id = f"M_{category.upper()}_S{slot:02d}"
        # En az 6 ay sürsün: kısa süren bir ödeme "düzenli" sayılmaz.
        duration = int(rng.integers(6, len(windows) + 1))

        for first, last in windows[-duration:]:
            day = int(np.clip(base_day + rng.integers(-3, 4), 1, calendar.monthrange(first.year, first.month)[1]))
            when = date(first.year, first.month, day)
            if when > last:      # kısmi son ayda henüz gelmemiş ödeme
                continue
            rows.append({
                "customer_id": customer_id,
                "transaction_date": datetime(when.year, when.month, when.day, 3, int(rng.integers(0, 60))),
                "amount": round(base_amount * float(rng.uniform(0.95, 1.05)), 2),
                "mcc_code": str(rng.choice(MCC_CODES[category])),
                "mcc_category": category,
                "channel": "mobil",
                "installment_count": 1,
                "merchant_id": merchant_id,
            })

    return rows


def generate_transactions(
    rng: np.random.Generator,
    customers: list[dict],
    latent: dict[str, dict],
    base_weights: dict[str, float],
    rate_by_segment: dict[str, int],
    windows: list[tuple[date, date]],
) -> tuple[list[dict], set[str]]:
    """Tüm işlemleri üretir. İkinci dönüş: sıçrama ekilen müşterilerin kimlikleri."""
    rows: list[dict] = []
    spiked: set[str] = set()

    merchants = {
        cat: [f"M_{cat.upper()}_{k:03d}" for k in range(MERCHANTS_PER_CATEGORY)]
        for cat in MCC_CODES
    }

    for customer in customers:
        cid = customer["customer_id"]
        lat = latent[cid]
        weights = _category_weights_for(lat["persona"], base_weights)
        monthly_rate = rate_by_segment[customer["customer_segment"]] * lat["activity"]

        # --- düzenli ödemeler (herkeste yok: negatif fixture lazım) ---
        if rng.random() < RECURRING_RATE:
            rows.extend(_make_recurring(rng, cid, windows))

        # --- normal harcama akışı ---
        for first, last in windows:
            # Kısmi son ayda işlem sayısı gün oranınca azalır.
            days_in_month = calendar.monthrange(first.year, first.month)[1]
            coverage = ((last - first).days + 1) / days_in_month
            n_tx = int(rng.poisson(monthly_rate * coverage))
            if n_tx == 0:
                continue

            # Mevsimsellik ayın kategori dağılımını değiştirir.
            month_weights = np.array([
                weights[cat] * SEASONALITY.get(cat, {}).get(first.month, 1.0)
                for cat in weights
            ])
            month_weights /= month_weights.sum()
            cats = rng.choice(list(weights), size=n_tx, p=month_weights)

            for category in cats:
                mu, sigma = CATEGORY_TICKET[category]
                amount = float(rng.lognormal(mu, sigma)) * lat["amount_factor"]
                installment = 1
                if (category in INSTALLMENT_CATEGORIES
                        and amount > INSTALLMENT_MIN_AMOUNT
                        and rng.random() < lat["installment_affinity"]):
                    installment = int(rng.choice(INSTALLMENT_CATEGORIES[category]))
                rows.append({
                    "customer_id": cid,
                    "transaction_date": _random_datetime(rng, first, last),
                    "amount": round(max(amount, 5.0), 2),
                    "mcc_code": str(rng.choice(MCC_CODES[category])),
                    "mcc_category": category,
                    "channel": _pick(rng, CHANNEL_WEIGHTS[category]),
                    "installment_count": installment,
                    "merchant_id": str(rng.choice(merchants[category])),
                })

        # --- olağandışı sıçrama (son 3 ay içinde tek seferlik büyük alım) ---
        if rng.random() < SPIKE_RATE:
            category = str(rng.choice(SPIKE_CATEGORIES))
            first, last = windows[-int(rng.integers(1, 4))]
            mu, sigma = CATEGORY_TICKET[category]
            amount = float(np.exp(mu)) * lat["amount_factor"] * float(rng.uniform(5.0, 12.0))
            rows.append({
                "customer_id": cid,
                "transaction_date": _random_datetime(rng, first, last),
                "amount": round(amount, 2),
                "mcc_code": str(rng.choice(MCC_CODES[category])),
                "mcc_category": category,
                "channel": _pick(rng, CHANNEL_WEIGHTS[category]),
                "installment_count": int(rng.choice([6, 9, 12])),
                "merchant_id": str(rng.choice(merchants[category])),
            })
            spiked.add(cid)

    rows.sort(key=lambda r: r["transaction_date"])
    for idx, row in enumerate(rows):
        row["transaction_id"] = f"T{idx:08d}"
        row["transaction_date"] = row["transaction_date"].isoformat()

    # Şema alan sırasına getir (JSON okunabilirliği için).
    order = list(Transaction.model_fields.keys())
    return [{k: r[k] for k in order} for r in rows], spiked


# --------------------------------------------------------------------------
# 3) Etkileşimler (model hedefi)
# --------------------------------------------------------------------------


def _is_eligible(customer: dict, campaign: Campaign) -> bool:
    """Geçmişte bu müşteriye bu kampanya sunulmuş olabilir mi.

    NOT: Bu, Gün 6'da yazılacak `agent.tools.check_eligibility`'nin basitleştirilmiş
    ikizidir ve amacı sadece geçmiş verinin tutarlı olmasıdır. Araç yazıldığında
    buradaki mantık ORADAN import edilmeli — iki yerde ayrı ayrı yaşamamalı.
    Bütçe kontrolü yok: teklif verildiği anda bütçe vardı.
    """
    if not customer["opt_in_marketing"]:
        return False        # KVKK: izinsiz müşteriye teklif sunulmaz, geçmişte de sunulmadı
    if campaign.eligible_segments and customer["customer_segment"] not in [
        s.value for s in campaign.eligible_segments
    ]:
        return False
    if any(not customer[product] for product in campaign.required_products):
        return False
    if not (campaign.min_age <= customer["age"] <= campaign.max_age):
        return False
    if customer["tenure_months"] < campaign.min_tenure_months:
        return False
    if campaign.requires_digital_active and not customer["digital_active"]:
        return False
    return True


def _campaign_attractiveness(campaign: Campaign) -> float:
    """Ödül tipleri arasında kabaca karşılaştırılabilir bir cazibe skoru."""
    value, unit = campaign.reward_value, campaign.reward_unit.value
    if unit == "yuzde":
        return value / (10.0 if campaign.reward_type.value == "cashback" else 20.0)
    if unit == "tutar":
        return min(value / 100.0, 2.0)
    return value / 9.0          # taksit sayısı


def generate_interactions(
    rng: np.random.Generator,
    customers: list[dict],
    latent: dict[str, dict],
    campaigns: list[Campaign],
    category_share: dict[str, dict[str, float]],
    monthly_category_spend: dict[str, dict[str, float]],
    as_of: date,
    history_start: date,
    label_noise: float,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Geçmiş teklifleri ve `accepted` etiketini üretir.

    Dönüş: kayıtlar, gözlemlenebilir sinyal skoru, etiketler. Son ikisi çalışma
    sonundaki "tavan AUC" raporunu üretmek için kullanılır.
    """
    by_customer = {c["customer_id"]: c for c in customers}

    attract_raw = {c.campaign_id: _campaign_attractiveness(c) for c in campaigns}
    attract_values = np.array(list(attract_raw.values()))
    attract_z = dict(zip(attract_raw, _zscore(attract_values)))

    # --- teklif adaylarını topla (etiket henüz yok) ---
    pending: list[dict] = []
    for customer in customers:
        eligible = [c for c in campaigns if _is_eligible(customer, c)]
        # Teklif penceresi geçmiş veri aralığıyla kesişmeyen kampanyaları ele.
        eligible = [
            c for c in eligible
            if max(c.valid_from, history_start) <= min(c.valid_to, as_of)
        ]
        if not eligible:
            continue

        n_offers = min(len(eligible), int(rng.integers(4, 10)))
        chosen = rng.choice(len(eligible), size=n_offers, replace=False)
        for idx in chosen:
            campaign = eligible[int(idx)]
            window_start = max(campaign.valid_from, history_start)
            window_end = min(campaign.valid_to, as_of)
            pending.append({
                "customer_id": customer["customer_id"],
                "campaign": campaign,
                "offered_at": _random_datetime(rng, window_start, window_end),
            })

    if not pending:
        return [], np.array([]), np.array([])

    # --- gözlemlenebilir sinyaller ---
    fit_raw = np.empty(len(pending))
    friction_raw = np.empty(len(pending))
    channel_match = np.empty(len(pending))

    for i, item in enumerate(pending):
        cid, campaign = item["customer_id"], item["campaign"]
        targets = [c.value for c in campaign.target_categories]
        shares = category_share.get(cid, {})
        fit_raw[i] = sum(shares.get(t, 0.0) for t in targets)

        # Asgari harcama sürtünmesi: kampanyanın min_spend'i, müşterinin o
        # kategorilerdeki aylık harcamasına göre ne kadar yüksek.
        monthly = sum(monthly_category_spend.get(cid, {}).get(t, 0.0) for t in targets)
        friction_raw[i] = min(campaign.min_spend / max(monthly, 1.0), 3.0)

        customer = by_customer[cid]
        digital_channels = {"push", "mobil_app", "email"}
        matched = (
            campaign.offer_channel.value in digital_channels
            if customer["digital_active"]
            else campaign.offer_channel.value == "sms"
        )
        channel_match[i] = 1.0 if matched else -1.0

    observable = (
        W_FIT * _zscore(fit_raw)
        + W_ATTRACT * np.array([attract_z[i["campaign"].campaign_id] for i in pending])
        + W_CHANNEL * channel_match
        - W_FRICTION * _zscore(friction_raw)
    )

    # --- gizli terim + gürültü: modelin ulaşamayacağı tavan buradan doğar ---
    hidden = (
        W_LATENT * np.array([latent[i["customer_id"]]["propensity"] for i in pending])
        + rng.normal(0.0, label_noise * NOISE_SCALE, size=len(pending))
    )

    probability = 1.0 / (1.0 + np.exp(-(INTERCEPT + observable + hidden)))
    accepted = (rng.random(len(pending)) < probability).astype(int)

    # --- kayıtları yaz ---
    rows: list[dict] = []
    for i, item in enumerate(pending):
        offered_at = item["offered_at"]
        is_accepted = bool(accepted[i])
        if is_accepted:
            responded_at = offered_at + timedelta(hours=float(rng.uniform(1, 24 * 14)))
        elif rng.random() < 0.30:
            responded_at = offered_at + timedelta(hours=float(rng.uniform(1, 24 * 14)))
        else:
            responded_at = None     # sessiz kalan müşteri — gerçek hayatta çoğunluk
        if responded_at is not None and responded_at.date() > as_of:
            responded_at = None

        rows.append({
            "interaction_id": "",
            "customer_id": item["customer_id"],
            "campaign_id": item["campaign"].campaign_id,
            "offered_at": offered_at,
            "offer_channel": item["campaign"].offer_channel.value,
            "accepted": is_accepted,
            "responded_at": responded_at,
        })

    rows.sort(key=lambda r: r["offered_at"])
    for idx, row in enumerate(rows):
        row["interaction_id"] = f"I{idx:07d}"
        row["offered_at"] = row["offered_at"].isoformat()
        row["responded_at"] = row["responded_at"].isoformat() if row["responded_at"] else None

    return rows, observable, accepted


# --------------------------------------------------------------------------
# Özellik yardımcıları (etiket üretimi için, özellik katmanından bağımsız)
# --------------------------------------------------------------------------


def _spending_profiles(
    transactions: list[dict], n_months: int
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Müşteri başına kategori payları ve aylık kategori harcaması.

    Bunlar Gün 3'teki özellik tablosunun basit bir öncüsüdür; burada sadece
    etiket üretmek için hesaplanıyorlar.
    """
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in transactions:
        totals[row["customer_id"]][row["mcc_category"]] += row["amount"]

    shares: dict[str, dict[str, float]] = {}
    monthly: dict[str, dict[str, float]] = {}
    for cid, by_category in totals.items():
        grand = sum(by_category.values())
        shares[cid] = {cat: amount / grand for cat, amount in by_category.items()}
        monthly[cid] = {cat: amount / n_months for cat, amount in by_category.items()}
    return shares, monthly


# --------------------------------------------------------------------------
# Doğrulama ve rapor
# --------------------------------------------------------------------------


def _validate_sample(records: list[dict], model, label: str, rng: np.random.Generator) -> None:
    """Üretilen kayıtların bir örneğini şemadan geçirir.

    Amaç: hatayı diske yazdıktan sonra `validate.py`'de değil, burada yakalamak.
    """
    if not records:
        raise ValueError(f"{label}: hiç kayıt üretilmedi")
    sample_idx = rng.choice(len(records), size=min(300, len(records)), replace=False)
    for idx in sample_idx:
        model.model_validate(records[int(idx)])


def _print_report(
    customers: list[dict],
    transactions: list[dict],
    interactions: list[dict],
    latent: dict[str, dict],
    spiked: set[str],
    base_weights: dict[str, float],
    observable: np.ndarray,
    accepted: np.ndarray,
    out_dir: Path,
) -> None:
    """Üretim sonrası özet — verinin gerçekçi olup olmadığı buradan okunur."""
    print("\n=== Sentetik Veri Üretim Raporu ===\n")
    print(f"  customers      {len(customers):>9,}")
    print(f"  transactions   {len(transactions):>9,}")
    print(f"  interactions   {len(interactions):>9,}")

    # -- işlem yoğunluğu (analyze_spending'in taşıyabilirliği) --
    per_customer: dict[str, int] = defaultdict(int)
    for row in transactions:
        per_customer[row["customer_id"]] += 1
    by_segment: dict[str, list[int]] = defaultdict(list)
    for customer in customers:
        by_segment[customer["customer_segment"]].append(per_customer[customer["customer_id"]])

    print("\n--- Müşteri başına işlem (yıllık / aylık) ---")
    for segment in ("mass", "affluent", "private"):
        values = by_segment.get(segment, [0])
        print(f"  {segment:<10} n={len(values):>5,}  yıllık ort {np.mean(values):>6.1f}  aylık ort {np.mean(values)/12:>5.1f}")

    # -- kategori dağılımı: config hedefine yakın mı --
    spend: dict[str, float] = defaultdict(float)
    for row in transactions:
        spend[row["mcc_category"]] += row["amount"]
    grand = sum(spend.values())
    print("\n--- Kategori payı (gerçekleşen / config hedefi) ---")
    for category in sorted(spend, key=spend.get, reverse=True):
        print(f"  {category:<18} {spend[category]/grand:>6.1%}   hedef {base_weights.get(category, 0):>5.1%}")

    # -- PFM sinyalleri --
    recurring_customers = {
        row["customer_id"] for row in transactions if "_S0" in row["merchant_id"]
    }
    print("\n--- PFM sinyalleri (analyze_spending bunları bulacak) ---")
    print(f"  düzenli ödemesi olan müşteri : {len(recurring_customers):>6,} "
          f"({len(recurring_customers)/len(customers):.0%})")
    print(f"  sıçrama ekilen müşteri       : {len(spiked):>6,} "
          f"({len(spiked)/len(customers):.0%})")

    # -- gizli yapı (diske yazılmadı) --
    persona_counts: dict[str, int] = defaultdict(int)
    for info in latent.values():
        persona_counts[info["persona"]] += 1
    print("\n--- Gizli persona dağılımı (VERİYE YAZILMADI) ---")
    for name, count in sorted(persona_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {count:>5,}  ({count/len(latent):.0%})")

    # -- etiket kalitesi --
    if len(accepted):
        rate = accepted.mean()
        ceiling = _auc(accepted, observable)
        print("\n--- Etiket kalitesi ---")
        print(f"  kabul oranı                   : {rate:.1%}   (sağlıklı bant %5-60)")
        print(f"  gözlemlenebilir sinyal tavanı : AUC {ceiling:.3f}   (hedef bant 0.72-0.82)")
        if not 0.05 <= rate <= 0.60:
            print("  [!] Kabul oranı bant dışı — INTERCEPT sabitini ayarla.")
        if not 0.72 <= ceiling <= 0.82:
            print("  [!] Tavan bant dışı — W_LATENT / label_noise ile gözlemlenebilir "
                  "katsayıların oranını değiştir.")

    print("\n--- Yazılan dosyalar ---")
    for name in ("customers.json", "transactions.json", "interactions.json"):
        path = out_dir / name
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("\nSıradaki adım: python -m src.data.validate\n")


# --------------------------------------------------------------------------
# Giriş noktası
# --------------------------------------------------------------------------


def _write_json(path: Path, records: list[dict], compact: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            records, f, ensure_ascii=False,
            separators=(",", ":") if compact else (", ", ": "),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sentetik kampanya/harcama verisi üretir.")
    parser.add_argument("--n-customers", type=int, default=None, help="config'teki değeri ezer")
    parser.add_argument("--seed", type=int, default=None, help="config'teki değeri ezer")
    parser.add_argument("--out", type=str, default=None, help="çıktı dizini (varsayılan data/raw)")
    parser.add_argument("--as-of", type=str, default=None, help="referans tarih, YYYY-MM-DD")
    args = parser.parse_args(argv)

    config = load_config()
    gen_cfg = config["generator"]

    seed = args.seed if args.seed is not None else gen_cfg["seed"]
    n_customers = args.n_customers if args.n_customers is not None else gen_cfg["n_customers"]
    n_months = gen_cfg["n_months"]
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    out_dir = resolve_path(args.out or config["data"]["json"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    windows = _month_windows(as_of, n_months)
    history_start = windows[0][0]

    print(f"Üretim başlıyor — seed={seed}, n_customers={n_customers:,}, "
          f"geçmiş {history_start} → {as_of}")

    # --- kampanya kataloğu: üretilmez, okunur ---
    catalog_path = out_dir / config["data"]["json"]["campaigns_file"]
    if not catalog_path.exists():
        print(f"HATA: kampanya kataloğu bulunamadı: {catalog_path}", file=sys.stderr)
        return 1
    with catalog_path.open(encoding="utf-8") as f:
        campaigns = [Campaign.model_validate(c) for c in json.load(f)]

    # --- üretim ---
    customers, latent = generate_customers(rng, n_customers, as_of, gen_cfg["opt_in_rate"])

    transactions, spiked = generate_transactions(
        rng, customers, latent,
        base_weights=gen_cfg["category_weights"],
        rate_by_segment=gen_cfg["transactions_per_month"],
        windows=windows,
    )

    shares, monthly = _spending_profiles(transactions, n_months)
    interactions, observable, accepted = generate_interactions(
        rng, customers, latent, campaigns, shares, monthly,
        as_of=as_of, history_start=history_start, label_noise=gen_cfg["label_noise"],
    )

    # --- diske yazmadan önce şema kontrolü ---
    _validate_sample(customers, Customer, "customers", rng)
    _validate_sample(transactions, Transaction, "transactions", rng)
    _validate_sample(interactions, Interaction, "interactions", rng)

    _write_json(out_dir / config["data"]["json"]["customers_file"], customers, compact=False)
    _write_json(out_dir / config["data"]["json"]["transactions_file"], transactions, compact=True)
    _write_json(out_dir / config["data"]["json"]["interactions_file"], interactions, compact=True)

    _print_report(
        customers, transactions, interactions, latent, spiked,
        gen_cfg["category_weights"], observable, accepted, out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
