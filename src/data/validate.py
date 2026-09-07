"""Veri kaynağı doğrulayıcı.

Bir veri kaynağının (JSON bugün, SQL yarın) sisteme bağlanabilmesi için bu
kontrolden geçmesi zorunludur. Amaç: gerçek veriye geçildiğinde hatanın
model eğitiminin ortasında değil, ilk adımda ve anlaşılır bir mesajla çıkması.

Kontroller:
  1. Şema uyumu   — zorunlu kolonlar var mı, örnek kayıtlar Pydantic'ten geçiyor mu
  2. Referans bütünlüğü — işlem/etkileşim kayıtları var olan müşteri ve kampanyaya mı bağlı
  3. PII sızıntısı — kaynakta olmaması gereken kişisel veri kolonları var mı

Çalıştırma:
    python -m src.data.validate
"""

from __future__ import annotations

import argparse

import sys
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from src.config import load_config
from src.data.schemas import TABLE_SCHEMAS, required_columns
from src.data.source import DataSource, build_data_source

#: Hiçbir veri kaynağında bulunmaması gereken kolon adları.
#: Bu liste bir güvenlik ağıdır: gerçek banka tablosuna bağlanırken yanlışlıkla
#: "SELECT *" yazılırsa doğrulama burada patlar, veri sisteme hiç girmez.
PII_COLUMNS = {
    "ad", "soyad", "adsoyad", "isim", "name", "first_name", "last_name", "full_name",
    "tckn", "tc_kimlik_no", "national_id", "vkn",
    "telefon", "phone", "gsm", "msisdn", "cep_telefonu",
    "email", "eposta", "e_posta", "mail",
    "iban", "hesap_no", "account_number",
    "kart_no", "card_number", "pan", "masked_pan",
    "adres", "address", "dogum_tarihi", "birth_date",
}

#: Tablo bazlı PII muafiyetleri. `campaigns.name` KAMPANYANIN adıdır, bir kişinin
#: adı değil. PII listesi bilinçli olarak jenerik ve geniş tutulduğu için, kişi
#: verisi taşımayan tablolarda bu tür çakışmalar tek tek muaf tutulur.
#: Buraya ekleme yaparken soru şu: "bu kolon bir GERÇEK KİŞİYİ işaret edebilir mi?"
PII_EXEMPT: dict[str, set[str]] = {
    "campaigns": {"name"},
}

#: Örnekleme boyutu — büyük tablolarda tüm satırları Pydantic'ten geçirmek pahalı.
SAMPLE_SIZE = 200


@dataclass
class ValidationReport:
    """Doğrulama sonucu."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def print_summary(self) -> None:
        print("\n=== Veri Kaynağı Doğrulama Raporu ===\n")

        for table, n in self.row_counts.items():
            print(f"  {table:<15} {n:>8,} satır")

        if self.warnings:
            print("\n--- Uyarılar ---")
            for w in self.warnings:
                print(f"  [!] {w}")

        if self.errors:
            print("\n--- HATALAR ---")
            for e in self.errors:
                print(f"  [X] {e}")
            print(f"\nSONUÇ: BAŞARISIZ ({len(self.errors)} hata)\n")
        else:
            print("\nSONUÇ: BAŞARILI — kaynak şemaya uygun.\n")


def _null_to_none(value):
    """pandas NaN/NaT değerlerini Python None'a çevirir.

    Gerekli çünkü JSON `null` ve SQL `NULL`, DataFrame'e girdiğinde tipe göre NaN
    veya NaT olur; Pydantic ise `float | None` alanında NaN'ı geçerli bir float
    sanıp `gt=0` kısıtına takılır (`Campaign.max_reward` bunun canlı örneği).
    Sessiz kabul de tehlikeli: `Interaction.responded_at` NaT olarak geçip tarih
    kontrolünü atlatıyordu.
    """
    if pd.api.types.is_list_like(value):
        return value            # target_categories gibi liste alanları
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _check_pii(df: pd.DataFrame, table: str, report: ValidationReport) -> None:
    """Kaynakta kişisel veri kolonu var mı."""
    exempt = PII_EXEMPT.get(table, set())
    found = {
        c for c in df.columns
        if c.lower().strip() in PII_COLUMNS and c.lower().strip() not in exempt
    }
    if found:
        report.errors.append(
            f"{table}: PII kolonu tespit edildi -> {sorted(found)}. "
            f"Bu kolonlar veri kaynağından çıkarılmalı (bkz. schemas.Customer)."
        )


def _check_schema(df: pd.DataFrame, table: str, report: ValidationReport) -> None:
    """Zorunlu kolonlar ve örnek kayıtların şema uyumu."""
    expected = set(required_columns(table))
    actual = set(df.columns)

    missing = expected - actual
    if missing:
        report.errors.append(f"{table}: eksik kolon(lar) -> {sorted(missing)}")

    extra = actual - expected
    if extra:
        report.warnings.append(
            f"{table}: şemada olmayan fazladan kolon(lar) -> {sorted(extra)} (yok sayılacak)"
        )

    if missing:
        return  # kolonlar eksikken kayıt doğrulamanın anlamı yok

    model = TABLE_SCHEMAS[table]
    sample = df.head(SAMPLE_SIZE)
    for idx, row in enumerate(sample.to_dict(orient="records")):
        payload = {k: _null_to_none(row[k]) for k in expected}
        try:
            model.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"])
            report.errors.append(
                f"{table}: {idx}. satır şemaya uymuyor -> alan '{loc}': {first['msg']}"
            )
            return  # ilk hata yeterli, rapor kalabalıklaşmasın


def _check_referential_integrity(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    campaigns: pd.DataFrame,
    interactions: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Yabancı anahtarlar gerçekten var olan kayıtlara mı işaret ediyor."""
    customer_ids = set(customers.get("customer_id", pd.Series(dtype=str)))
    campaign_ids = set(campaigns.get("campaign_id", pd.Series(dtype=str)))

    orphan_txn = set(transactions.get("customer_id", pd.Series(dtype=str))) - customer_ids
    if orphan_txn:
        report.errors.append(
            f"transactions: {len(orphan_txn)} adet bilinmeyen customer_id "
            f"(örn. {sorted(orphan_txn)[:3]})"
        )

    orphan_int_cust = set(interactions.get("customer_id", pd.Series(dtype=str))) - customer_ids
    if orphan_int_cust:
        report.errors.append(
            f"interactions: {len(orphan_int_cust)} adet bilinmeyen customer_id "
            f"(örn. {sorted(orphan_int_cust)[:3]})"
        )

    orphan_int_camp = set(interactions.get("campaign_id", pd.Series(dtype=str))) - campaign_ids
    if orphan_int_camp:
        report.errors.append(
            f"interactions: {len(orphan_int_camp)} adet bilinmeyen campaign_id "
            f"(örn. {sorted(orphan_int_camp)[:3]})"
        )

    # Modelin öğrenebilmesi için hedef değişkende iki sınıf da bulunmalı.
    if "accepted" in interactions.columns and len(interactions):
        rate = float(interactions["accepted"].mean())
        if rate in (0.0, 1.0):
            report.errors.append(
                f"interactions: 'accepted' tek sınıftan oluşuyor (oran={rate:.2f}) — "
                f"model eğitilemez."
            )
        elif not 0.05 <= rate <= 0.60:
            report.warnings.append(
                f"interactions: kabul oranı {rate:.1%} — gerçekçi bant genelde %5-40'tır. "
                f"generator.label_noise / cazibe parametrelerini gözden geçir."
            )


def validate_source(source: DataSource) -> ValidationReport:
    """Bir veri kaynağını baştan sona doğrular ve rapor döner."""
    report = ValidationReport()

    try:
        tables = {
            "customers": source.get_customers(),
            "transactions": source.get_transactions(),
            "campaigns": source.get_campaigns(),
            "interactions": source.get_interactions(),
        }
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        report.errors.append(f"Veri okunamadı: {exc}")
        return report

    for name, df in tables.items():
        report.row_counts[name] = len(df)
        _check_pii(df, name, report)
        _check_schema(df, name, report)

    _check_referential_integrity(
        tables["customers"],
        tables["transactions"],
        tables["campaigns"],
        tables["interactions"],
        report,
    )

    return report


def main(argv: list[str] | None = None) -> int:
    # Bkz. features/build.py: parser olmadan --help doğrulamayı çalıştırıyordu.
    argparse.ArgumentParser(
        description="Aktif veri kaynağını şemaya karşı doğrular."
    ).parse_args(argv)

    config = load_config()
    source = build_data_source(config)
    report = validate_source(source)
    report.print_summary()
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
