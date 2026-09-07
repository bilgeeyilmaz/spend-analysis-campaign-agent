"""Veri sözleşmesi (data contract).

Bu dosya projenin en kritik parçasıdır: aşağıdaki her katman (özellik üretimi,
model, agent, API) SADECE bu şemaya bağlıdır — verinin nereden geldiğine değil.

Bugün veri sentetik JSON'dan geliyor; yarın maskelenmiş banka tablolarından
gelecek. O geçişte değişen tek şey `source.py` içindeki bir sınıf olur, bu şema
değişmez. Yeni bir veri kaynağının geçerli sayılması için `validate.py`
içindeki `validate_source()` kontrolünden geçmesi zorunludur.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Sabit değer kümeleri (enum)
# --------------------------------------------------------------------------


class MccCategory(str, Enum):
    """Harcama kategorisi.

    Gerçekte kart işlemleri 4 haneli MCC koduyla gelir (örn. 5411 = market).
    Modelde ham MCC yerine kategori kullanıyoruz: hem yorumlanabilir, hem de
    binlerce seyrek MCC yerine 12 dolu sınıf öğrenmek çok daha sağlıklı.
    Ham kod `Transaction.mcc_code` alanında saklanır, kaybolmaz.
    """

    MARKET = "market"
    AKARYAKIT = "akaryakit"
    RESTORAN = "restoran"
    GIYIM = "giyim"
    ELEKTRONIK = "elektronik"
    SEYAHAT = "seyahat"
    SAGLIK = "saglik"
    EGITIM = "egitim"
    EGLENCE = "eglence"
    TELEKOM = "telekom"
    ONLINE_ALISVERIS = "online_alisveris"
    DIGER = "diger"


class Channel(str, Enum):
    """İşlemin gerçekleştiği kanal."""

    POS = "pos"
    ONLINE = "online"
    ATM = "atm"
    MOBIL = "mobil"


class CustomerSegment(str, Enum):
    """Bankanın kendi müşteri segmenti — bizim ML segmentasyonumuzdan farklıdır.

    Bu alan bankadan hazır gelen ticari segmenttir. Gün 4'te KMeans ile
    üreteceğimiz davranışsal segment ayrı bir alandır. İkisini karıştırma:
    sunumda "zaten segment var, niye model kuruyorsun?" sorusunun cevabı
    tam olarak bu ayrımdır.
    """

    MASS = "mass"
    AFFLUENT = "affluent"
    PRIVATE = "private"


class IncomeBand(str, Enum):
    """Gelir bandı — ham gelir yerine bantlanmış hali.

    Bankalar ham geliri modele nadiren açar; maskelenmiş veride de bant gelir.
    Sentetik veriyi baştan bantlı üretmek, gerçek veriye geçişi kolaylaştırır.
    A = en yüksek, D = en düşük.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class RewardType(str, Enum):
    """Kampanyanın müşteriye sunduğu fayda tipi."""

    CASHBACK = "cashback"          # nakit iade
    PUAN = "puan"                  # sadakat puanı
    INDIRIM = "indirim"            # doğrudan indirim
    TAKSIT = "taksit"              # ek taksit imkânı


class RewardUnit(str, Enum):
    """`reward_value` alanının birimi."""

    YUZDE = "yuzde"                # %10 iade
    TUTAR = "tutar"                # 250 TL iade
    TAKSIT_SAYISI = "taksit_sayisi"  # +3 taksit


class OfferChannel(str, Enum):
    """Kampanyanın müşteriye ulaştırılacağı kanal."""

    SMS = "sms"
    PUSH = "push"
    EMAIL = "email"
    MOBIL_APP = "mobil_app"


# --------------------------------------------------------------------------
# Ana varlıklar
# --------------------------------------------------------------------------


class Customer(BaseModel):
    """Müşteri ana kaydı.

    DİKKAT — PII politikası: bu şemada bilinçli olarak ad, soyad, TCKN, telefon,
    e-posta, IBAN, kart numarası YOKTUR. Sentetik veride de üretilmez. Böylece
    "yanlışlıkla LLM'e PII gitmesi" riski şema seviyesinde ortadan kalkar.
    Gerçek veriye geçişte `SqlDataSource` bu alanları SELECT etmemelidir.
    """

    customer_id: str = Field(..., description="Benzersiz müşteri anahtarı, örn. C000123")
    age: int = Field(..., ge=18, le=100)
    gender: str | None = Field(
        default=None,
        description=(
            "Sadece veri bütünlüğü için tutulur. Hedefleme özelliği olarak "
            "KULLANILMAZ (bkz. features/build.py EXCLUDED_FEATURES) — "
            "cinsiyete dayalı fiyat/teklif ayrımı yasal risk taşır."
        ),
    )
    city: str
    customer_segment: CustomerSegment
    income_band: IncomeBand
    tenure_months: int = Field(..., ge=0, description="Bankayla çalışma süresi (ay)")

    # Ürün sahipliği — kampanya uygunluk kurallarının temel girdisi
    has_credit_card: bool
    has_debit_card: bool
    has_loan: bool
    has_deposit: bool

    digital_active: bool = Field(
        ..., description="Son 30 günde mobil/internet bankacılığına giriş yaptı mı"
    )
    opt_in_marketing: bool = Field(
        ...,
        description=(
            "KVKK ticari elektronik ileti izni. False ise müşteriye kampanya "
            "İLETİLEMEZ — bu bir model kararı değil, hukuki zorunluluktur ve "
            "check_eligibility içinde ilk kontrol edilen kuraldır."
        ),
    )
    created_at: date = Field(..., description="Müşteri olma tarihi")

    @field_validator("customer_id")
    @classmethod
    def _id_bos_olamaz(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("customer_id boş olamaz")
        return v.strip()


class Transaction(BaseModel):
    """Tek bir kart/hesap işlemi.

    Özellik üretiminin (RFM, kategori payları, trend) tek kaynağıdır.
    """

    transaction_id: str
    customer_id: str
    transaction_date: datetime
    amount: float = Field(..., gt=0, description="İşlem tutarı (TRY), her zaman pozitif")
    mcc_code: str = Field(..., description="Ham 4 haneli MCC kodu, örn. '5411'")
    mcc_category: MccCategory = Field(..., description="MCC'nin eşlendiği kategori")
    channel: Channel
    installment_count: int = Field(
        default=1, ge=1, le=12, description="1 = tek çekim, >1 = taksitli"
    )
    merchant_id: str


class Campaign(BaseModel):
    """Kampanya kataloğu kaydı.

    Propensity modeli girdisi = müşteri özellikleri ⊕ KAMPANYA özellikleri.
    Bu sayede kataloğa yeni kampanya eklendiğinde modeli yeniden eğitmek
    gerekmez — canlı demoda "hadi yeni kampanya ekleyelim" anını mümkün
    kılan tasarım budur.
    """

    campaign_id: str
    name: str
    description: str

    target_categories: list[MccCategory] = Field(
        ..., min_length=1, description="Kampanyanın hedeflediği harcama kategorileri"
    )
    reward_type: RewardType
    reward_value: float = Field(..., gt=0)
    reward_unit: RewardUnit
    max_reward: float | None = Field(
        default=None, gt=0, description="Üst sınır (TL). Yüzde tipli kampanyalarda anlamlı."
    )
    min_spend: float = Field(default=0.0, ge=0, description="Kampanya için asgari harcama (TL)")

    valid_from: date
    valid_to: date
    total_budget: float = Field(..., gt=0)
    remaining_budget: float = Field(..., ge=0)

    # --- Uygunluk kuralları (check_eligibility bunları okur) ---
    required_products: list[str] = Field(
        default_factory=list,
        description="Gerekli ürünler: has_credit_card / has_loan / has_deposit / has_debit_card",
    )
    eligible_segments: list[CustomerSegment] = Field(default_factory=list)
    min_tenure_months: int = Field(default=0, ge=0)
    min_age: int = Field(default=18, ge=18)
    max_age: int = Field(default=100, le=100)
    requires_digital_active: bool = False

    offer_channel: OfferChannel
    priority: int = Field(default=5, ge=1, le=10, description="1 = en yüksek öncelik")

    @model_validator(mode="after")
    def _tarih_ve_butce_tutarli(self) -> Campaign:
        if self.valid_to < self.valid_from:
            raise ValueError(
                f"{self.campaign_id}: valid_to ({self.valid_to}) "
                f"valid_from'dan ({self.valid_from}) önce olamaz"
            )
        if self.remaining_budget > self.total_budget:
            raise ValueError(
                f"{self.campaign_id}: remaining_budget ({self.remaining_budget}) "
                f"total_budget'ı ({self.total_budget}) aşamaz"
            )
        if self.max_age < self.min_age:
            raise ValueError(f"{self.campaign_id}: max_age, min_age'den küçük olamaz")
        return self

    def is_active(self, as_of: date) -> bool:
        """Kampanya verilen tarihte yayında mı (tarih + bütçe)."""
        return self.valid_from <= as_of <= self.valid_to and self.remaining_budget > 0


class Interaction(BaseModel):
    """Geçmişte sunulmuş bir kampanya teklifi ve müşterinin tepkisi.

    `accepted` alanı propensity modelinin HEDEF değişkenidir.
    Sentetik veride bu etiket, gizli müşteri eğilimi + kategori uyumu +
    gürültü ile üretilir (bkz. generator.py) — basit bir kuralla değil,
    yoksa model kuralı ezberler ve metrikler yapay şekilde mükemmel çıkar.
    """

    interaction_id: str
    customer_id: str
    campaign_id: str
    offered_at: datetime
    offer_channel: OfferChannel
    accepted: bool = Field(..., description="MODEL HEDEFİ: müşteri kampanyayı kullandı mı")
    responded_at: datetime | None = None

    @model_validator(mode="after")
    def _cevap_tarihi_tekliften_sonra(self) -> Interaction:
        if self.responded_at is not None and self.responded_at < self.offered_at:
            raise ValueError(
                f"{self.interaction_id}: responded_at, offered_at'ten önce olamaz"
            )
        return self


# --------------------------------------------------------------------------
# Sözleşme kaydı — validate.py ve source.py bu tabloyu kullanır
# --------------------------------------------------------------------------

#: Mantıksal tablo adı -> onu doğrulayan Pydantic modeli.
#: Yeni bir tablo eklendiğinde buraya da eklenmeli; validate_source() bu
#: sözlüğü gezerek her kaynağı aynı kurallarla sınar.
TABLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "customers": Customer,
    "transactions": Transaction,
    "campaigns": Campaign,
    "interactions": Interaction,
}


def required_columns(table: str) -> list[str]:
    """Bir tablonun zorunlu kolon listesi.

    `validate_source()` ve `SqlDataSource.column_map` doğrulaması bunu kullanır.
    """
    if table not in TABLE_SCHEMAS:
        raise KeyError(f"Bilinmeyen tablo: {table!r}. Geçerli: {list(TABLE_SCHEMAS)}")
    return list(TABLE_SCHEMAS[table].model_fields.keys())
