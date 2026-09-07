"""Agent'ın araç katmanı — beş aracın saf Python implementasyonu.

    from src.agent.tools import ToolContext
    ctx = ToolContext()
    ctx.analyze_spending("C000031")

Bu dosyada LLM YOKTUR ve olmayacak. Araçlar tek başına çağrılabilir, tek başına
test edilir ve `GROQ_API_KEY` olmadan da çalışır. Gün 7'deki orkestratör bunların
üstüne oturur; sıra tersine çevrilirse hata ayıklanamaz bir sistem çıkar — model
mi yanlış, araç mı yanlış, LLM mi yanlış anlaşılmaz.

Beş araç:

    get_customer_profile   müşteri künyesi (PII'siz)
    analyze_spending       harcama analizi — kataloğa BAKMAZ, bkz. spending.py
    score_campaigns        uygun kampanyaları modelle skorlar
    check_eligibility      iş kuralları; `opt_in_marketing` İLK kontrol
    get_campaign_details   katalog bilgisi

İki kural araçların içine gömülüdür, çağıranın nezaketine bırakılmaz:

  * **`opt_in_marketing` ilk kapıdır.** `check_eligibility` başka hiçbir kurala
    bakmadan önce izni kontrol eder ve izin yoksa orada durur. Hiçbir model
    skoru bu kapıyı açamaz. `score_campaigns` da uygunluğu kendi içinde
    uygular — LLM'in "uygunluğa bakmayı unutması" diye bir ihtimal kalmasın.
  * **Analiz tekliften önce gelir.** `analyze_spending` kampanya kataloğunu
    okumaz ve uygun kampanya olmasa bile anlamlı cevap üretir.

PII: `get_customer_profile` `gender` ve `city` alanlarını döndürmez. Şemada zaten
ad/TCKN/telefon/IBAN yok (bkz. `schemas.Customer`), yani sızdıracak alan
strüktürel olarak mevcut değil; buradaki eleme ikinci ağ. Üçüncüsü Gün 8'de
`guardrails.mask_payload()`.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import load_config, resolve_path
from src.data.source import build_data_source
from src.agent import spending

#: `get_customer_profile` çıktısına giren alanlar. Liste beyaz listedir:
#: şemaya yeni bir alan eklenince otomatik sızmasın diye açıkça sayılıyor.
PROFILE_FIELDS: list[str] = [
    "customer_id",
    "age",
    "customer_segment",
    "income_band",
    "tenure_months",
    "has_credit_card",
    "has_debit_card",
    "has_loan",
    "has_deposit",
    "digital_active",
    "opt_in_marketing",
]

#: Profil çıktısından KASTEN dışlananlar; testle bağlıdır.
PROFILE_EXCLUDED: frozenset[str] = frozenset({"gender", "city", "created_at"})


class ToolContext:
    """Araçların paylaştığı veri/model erişimi.

    Veri kaynağı, özellik tablosu ve propensity paketi ilk kullanımda yüklenir
    ve örnek boyunca bellekte kalır. Her araç çağrısında 80 MB JSON okumak
    demoyu kullanılamaz hâle getirirdi.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.source = build_data_source(self.config)
        self._features: pd.DataFrame | None = None
        self._propensity: dict | None = None
        self._segments = None
        self._as_of: pd.Timestamp | None = None

    # -- tembel yüklemeler --------------------------------------------------

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            path = resolve_path(self.config["data"]["processed_dir"]) / "customer_features.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{path} yok. Önce: python -m src.features.build")
            self._features = pd.read_parquet(path)
        return self._features

    @property
    def segments(self) -> pd.DataFrame | None:
        """Davranışsal segment tablosu; üretilmemişse None.

        Segment bir **sunum** bilgisidir: araç sözleşmesine (`PROFILE_FIELDS`)
        eklenmedi, çünkü LLM'in eline geçen her alan cevap metnine sızabilir ve
        "sizi şu segmente koyduk" cümlesi müşteriye söylenecek bir şey değil.
        API künyeye ekler, arayüz rozet olarak gösterir.
        """
        if self._segments is None:
            path = resolve_path(self.config["data"]["processed_dir"]) / "customer_segments.parquet"
            self._segments = pd.read_parquet(path) if path.exists() else False
        return self._segments if self._segments is not False else None

    def get_behavior_segment(self, customer_id: str) -> str | None:
        df = self.segments
        if df is None:
            return None
        satir = df[df["customer_id"] == customer_id]
        return None if satir.empty else str(satir.iloc[0]["behavior_segment"])

    @property
    def propensity(self) -> dict:
        if self._propensity is None:
            from src.models.propensity import load_package

            self._propensity = load_package(self.config)
        return self._propensity

    @property
    def as_of(self) -> pd.Timestamp:
        """Uygunluk kontrolünün referans tarihi.

        `config.features.as_of_date` doluysa o; değilse verideki en son işlem
        tarihi. Bugünün tarihini kullanmak, sentetik veri birkaç gün eskidiğinde
        kampanyaları sessizce "süresi geçmiş" yapar ve demo bozulur.
        """
        if self._as_of is None:
            ayar = self.config["features"].get("as_of_date")
            self._as_of = (
                pd.Timestamp(ayar) if ayar
                else pd.to_datetime(self.source.get_transactions()["transaction_date"]).max()
            )
        return self._as_of

    # -- yardımcılar --------------------------------------------------------

    def _customer(self, customer_id: str) -> pd.Series:
        musteriler = self.source.get_customers()
        satir = musteriler[musteriler["customer_id"] == customer_id]
        if satir.empty:
            raise KeyError(f"Müşteri bulunamadı: {customer_id}")
        return satir.iloc[0]

    def _campaign(self, campaign_id: str) -> pd.Series:
        katalog = self.source.get_campaigns()
        satir = katalog[katalog["campaign_id"] == campaign_id]
        if satir.empty:
            raise KeyError(f"Kampanya bulunamadı: {campaign_id}")
        return satir.iloc[0]

    # -- 1. profil ----------------------------------------------------------

    def get_customer_profile(self, customer_id: str) -> dict:
        """Müşteri künyesi. PII içermez (bkz. modül başlığı)."""
        musteri = self._customer(customer_id)
        profil = {
            alan: musteri[alan] for alan in PROFILE_FIELDS if alan in musteri.index
        }
        # numpy tiplerini JSON'a çevrilebilir hâle getir
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in profil.items()}

    # -- 2. harcama analizi -------------------------------------------------

    def analyze_spending(self, customer_id: str, months: int | None = None) -> dict:
        """Harcama analizi. Kampanya kataloğunu OKUMAZ.

        Tek başına anlamlı bir cevap üretir; öneri bu çıktının üstüne oturur.
        Ayrıntı ve eşikler için `src/agent/spending.py`.
        """
        self._customer(customer_id)          # yoksa KeyError
        tx = self.source.get_transactions(customer_id=customer_id)
        sonuc = spending.analiz_et(
            tx, as_of=self.as_of, months=months,
            ayar=self.config["agent"]["spending_analysis"],
        )
        return {"customer_id": customer_id, **sonuc}

    # -- 3. uygunluk --------------------------------------------------------

    def check_eligibility(self, customer_id: str, campaign_id: str) -> dict:
        """İş kuralları kontrolü. Kararı model değil, bu fonksiyon verir.

        `opt_in_marketing` İLK ve tek başına belirleyici kontroldür: izin yoksa
        diğer kurallara hiç bakılmaz. Bu bir performans optimizasyonu değil,
        hukuki bir sıralamadır — izin bir sinyal değil, ön koşuldur.
        """
        musteri = self._customer(customer_id)
        kampanya = self._campaign(campaign_id)
        bugun = self.as_of.date()

        if not bool(musteri["opt_in_marketing"]):
            return {
                "customer_id": customer_id, "campaign_id": campaign_id,
                "uygun": False,
                "basarisiz_kurallar": ["opt_in_marketing"],
                "gerekce": "Müşterinin ticari elektronik ileti izni yok; "
                           "kampanya teklifi yapılamaz.",
            }

        basarisiz: list[str] = []

        valid_from, valid_to = kampanya["valid_from"], kampanya["valid_to"]
        if not (_as_date(valid_from) <= bugun <= _as_date(valid_to)):
            basarisiz.append("kampanya_tarihi")
        if float(kampanya["remaining_budget"]) <= 0:
            basarisiz.append("butce_tukendi")

        segmentler = list(kampanya["eligible_segments"] or [])
        if segmentler and musteri["customer_segment"] not in segmentler:
            basarisiz.append("segment")

        for urun in list(kampanya["required_products"] or []):
            if not bool(musteri.get(urun, False)):
                basarisiz.append(f"urun:{urun}")

        if int(musteri["tenure_months"]) < int(kampanya["min_tenure_months"]):
            basarisiz.append("kidem")
        if not (int(kampanya["min_age"]) <= int(musteri["age"]) <= int(kampanya["max_age"])):
            basarisiz.append("yas")
        if bool(kampanya["requires_digital_active"]) and not bool(musteri["digital_active"]):
            basarisiz.append("dijital_aktiflik")

        return {
            "customer_id": customer_id,
            "campaign_id": campaign_id,
            "uygun": not basarisiz,
            "basarisiz_kurallar": basarisiz,
            "gerekce": (
                "Müşteri kampanyanın tüm koşullarını sağlıyor."
                if not basarisiz
                else "Sağlanmayan koşul(lar): " + ", ".join(basarisiz)
            ),
        }

    # -- 4. skorlama --------------------------------------------------------

    def score_campaigns(self, customer_id: str, top_k: int | None = None) -> dict:
        """Uygun kampanyaları modelle skorlar, azalan sırada döner.

        Uygunluk filtresi BURADA uygulanır. LLM'in `check_eligibility`'yi
        çağırmayı atlaması ihtimaline karşı: uygun olmayan bir kampanya bu
        aracın çıktısına hiç girmez, dolayısıyla önerilemez.
        """
        from src.models.propensity import score_campaigns as model_skorla

        self._customer(customer_id)
        top_k = top_k or self.config["agent"]["top_k_campaigns"]
        katalog = self.source.get_campaigns()

        uygunluk = {
            cid: self.check_eligibility(customer_id, cid)
            for cid in katalog["campaign_id"]
        }
        uygun_idler = [cid for cid, u in uygunluk.items() if u["uygun"]]
        if not uygun_idler:
            elenme = sorted({
                kural for u in uygunluk.values() for kural in u["basarisiz_kurallar"]
            })
            return {
                "customer_id": customer_id, "oneriler": [],
                "uygun_kampanya_sayisi": 0,
                "gerekce": "Uygun kampanya yok. Elenme nedenleri: " + ", ".join(elenme),
            }

        musteri_ozellik = self.features[self.features["customer_id"] == customer_id]
        if musteri_ozellik.empty:
            raise KeyError(f"{customer_id} özellik tablosunda yok — src.features.build çalıştır.")

        skorlar = model_skorla(
            self.propensity, musteri_ozellik,
            katalog[katalog["campaign_id"].isin(uygun_idler)],
        )
        isimler = katalog.set_index("campaign_id")["name"]

        oneriler = [
            {
                "campaign_id": satir["campaign_id"],
                "name": str(isimler[satir["campaign_id"]]),
                "score": round(float(satir["score"]), 4),
                "sira": i + 1,
            }
            for i, (_, satir) in enumerate(skorlar.head(top_k).iterrows())
        ]
        return {
            "customer_id": customer_id,
            "oneriler": oneriler,
            "uygun_kampanya_sayisi": len(uygun_idler),
            "model_versiyonu": self.propensity.get("trained_at"),
        }

    # -- 5. katalog ---------------------------------------------------------

    def get_campaign_details(self, campaign_id: str) -> dict:
        """Katalog kaydı. Tarihler ISO string'e çevrilir (JSON uyumu)."""
        kampanya = self._campaign(campaign_id)
        cikti = kampanya.to_dict()
        for alan in ("valid_from", "valid_to"):
            cikti[alan] = _as_date(cikti[alan]).isoformat()
        cikti["aktif"] = (
            _as_date(kampanya["valid_from"]) <= self.as_of.date() <= _as_date(kampanya["valid_to"])
            and float(kampanya["remaining_budget"]) > 0
        )
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in cikti.items()}

    # -- dağıtıcı -----------------------------------------------------------

    def call(self, name: str, arguments: dict) -> dict:
        """Ada göre araç çağırır. Gün 7'deki tool-calling döngüsü bunu kullanır."""
        if name not in TOOL_NAMES:
            raise ValueError(f"Bilinmeyen araç: {name!r}. Geçerli olanlar: {sorted(TOOL_NAMES)}")
        return getattr(self, name)(**arguments)


def _as_date(deger) -> date:
    """`date` | `datetime` | str -> `date`. JsonDataSource date döndürüyor ama
    SqlDataSource'un timestamp döndürmesi çok muhtemel."""
    if isinstance(deger, date) and not hasattr(deger, "hour"):
        return deger
    return pd.Timestamp(deger).date()


# --------------------------------------------------------------------------
# LLM'e verilecek imzalar (Gün 7)
# --------------------------------------------------------------------------

#: OpenAI/Groq tool-calling formatı. Açıklamalar Türkçe: modelin hangi aracı ne
#: zaman çağıracağına karar verdiği tek bilgi bu metinlerdir, sistem promptu
#: değil. `analyze_spending`'in açıklaması bilinçli olarak "tek başına yeterli"
#: der — serbest PFM sorusunda gereksiz yere kampanya skorlanmasın diye.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": "Müşterinin künyesi: yaş, segment, gelir bandı, kıdem, "
                           "ürün sahipliği, dijital aktiflik, pazarlama izni.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_spending",
            "description": "Müşterinin harcama analizi: kategori dağılımı, aylık seri, "
                           "trend, düzenli (abonelik benzeri) ödemeler ve olağandışı "
                           "harcama artışları. Harcamaya dair sorular için TEK BAŞINA "
                           "yeterlidir; kampanya sorulmadıkça başka araç çağırma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "months": {
                        "type": "integer",
                        "description": "Kategori kırılımının kaç aylık olacağı (varsayılan 3).",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_campaigns",
            "description": "Müşteri için UYGUN kampanyaları eğilim modeliyle skorlar ve "
                           "en iyiden başlayarak sıralar. Uygunluk kontrolü içeride "
                           "uygulanır; uygun olmayan kampanya sonuçta yer almaz.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "top_k": {"type": "integer", "description": "Kaç öneri döneceği."},
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Bir müşterinin belirli bir kampanyaya uygun olup olmadığını "
                           "iş kurallarıyla belirler (izin, tarih, bütçe, segment, ürün, "
                           "kıdem, yaş, dijital aktiflik). Uygunluk kararı yalnızca bu "
                           "aracındır; kendi başına uygunluk yorumu yapma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "campaign_id": {"type": "string"},
                },
                "required": ["customer_id", "campaign_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_campaign_details",
            "description": "Kampanyanın katalog bilgisi: ad, açıklama, hedef kategoriler, "
                           "ödül tipi ve değeri, asgari harcama, geçerlilik tarihleri.",
            "parameters": {
                "type": "object",
                "properties": {"campaign_id": {"type": "string"}},
                "required": ["campaign_id"],
            },
        },
    },
]

TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)
