"""Veri kaynağı soyutlaması.

"Sentetik veriyle başlayıp sonra gerçek veriye geçsek sorun olur mu?" sorusunun
mühendislik cevabı bu dosyadır: üst katmanlar `DataSource` arayüzünü çağırır,
verinin JSON dosyasından mı yoksa banka tablosundan mı geldiğini bilmez.

Gerçek veriye geçiş adımları:
  1. `SqlDataSource` içindeki sorguları ve `column_map`'i doldur.
  2. `config.yaml` içinde `data.source: sql` yap.
  3. `python -m src.data.validate` çalıştır — şema uyumu doğrulanır.
Özellik üretimi, model, agent ve API katmanlarına DOKUNULMAZ.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from src.config import resolve_path
from src.data.schemas import TABLE_SCHEMAS, required_columns


@runtime_checkable
class DataSource(Protocol):
    """Tüm veri kaynaklarının uyması gereken arayüz.

    Dönen DataFrame'lerin kolonları `schemas.py` içindeki ilgili modelin
    alan adlarıyla birebir aynı olmalıdır. Bu kural `validate_source()`
    tarafından zorlanır.
    """

    def get_customers(self) -> pd.DataFrame:
        """Tüm müşteriler."""
        ...

    def get_transactions(
        self,
        customer_id: str | None = None,
        since: date | None = None,
    ) -> pd.DataFrame:
        """İşlemler; opsiyonel olarak müşteriye ve tarihe göre filtrelenmiş.

        Filtreler arayüzün parçasıdır çünkü gerçek veride tüm işlem tablosunu
        belleğe çekmek mümkün olmaz — SQL kaynağı bu filtreleri WHERE'e çevirir.
        """
        ...

    def get_campaigns(
        self,
        active_only: bool = False,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Kampanya kataloğu. `active_only` ise tarih ve bütçe filtresi uygulanır."""
        ...

    def get_interactions(self) -> pd.DataFrame:
        """Geçmiş teklif/tepki kayıtları (model hedefi burada)."""
        ...


class JsonDataSource:
    """Sentetik JSON dosyalarından okuyan kaynak (staj boyunca varsayılan).

    Veri boyutu küçük olduğu için dosyalar bir kez okunup bellekte tutulur.
    """

    def __init__(self, raw_dir: str | Path, filenames: dict[str, str] | None = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.filenames = filenames or {
            "customers": "customers.json",
            "transactions": "transactions.json",
            "campaigns": "campaigns.json",
            "interactions": "interactions.json",
        }
        self._cache: dict[str, pd.DataFrame] = {}

    # -- iç yardımcılar ----------------------------------------------------

    def _load(self, table: str) -> pd.DataFrame:
        """Bir tabloyu diskten oku, tarih kolonlarını dönüştür, önbelleğe al."""
        if table in self._cache:
            return self._cache[table]

        path = self.raw_dir / self.filenames[table]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} bulunamadı. Önce sentetik veriyi üret:\n"
                f"    python -m src.data.generator"
            )

        with path.open(encoding="utf-8") as f:
            records = json.load(f)

        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError(f"{path} boş — generator hatalı çalışmış olabilir.")

        # Şemada tarih/zaman olan alanları gerçek datetime'a çevir.
        # (JSON'da hepsi string olarak durur.)
        for col in ("transaction_date", "offered_at", "responded_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in ("created_at", "valid_from", "valid_to"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        missing = set(required_columns(table)) - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} şemaya uymuyor. Eksik kolonlar: {sorted(missing)}"
            )

        self._cache[table] = df
        return df

    # -- DataSource arayüzü ------------------------------------------------

    def get_customers(self) -> pd.DataFrame:
        return self._load("customers").copy()

    def get_transactions(
        self,
        customer_id: str | None = None,
        since: date | None = None,
    ) -> pd.DataFrame:
        df = self._load("transactions")
        if customer_id is not None:
            df = df[df["customer_id"] == customer_id]
        if since is not None:
            df = df[df["transaction_date"] >= pd.Timestamp(since)]
        return df.copy()

    def get_campaigns(
        self,
        active_only: bool = False,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        df = self._load("campaigns")
        if active_only:
            ref = as_of or datetime.now().date()
            df = df[
                (df["valid_from"] <= ref)
                & (df["valid_to"] >= ref)
                & (df["remaining_budget"] > 0)
            ]
        return df.copy()

    def get_interactions(self) -> pd.DataFrame:
        return self._load("interactions").copy()


class SqlDataSource:
    """Gerçek / maskelenmiş banka verisi için iskelet.

    HENÜZ UYGULANMADI — veri erişimi açıldığında doldurulacak. Sınıfın burada
    durmasının sebebi, geçişin ne kadar dar bir yüzey olduğunu göstermektir:
    dört sorgu ve bir kolon eşlemesi. Üst katmanlarda hiçbir değişiklik yok.

    Uygulama notları:
      * `column_map`: banka kolon adı -> şema alan adı (config.yaml'dan gelir).
      * PII kolonları (ad, TCKN, telefon, IBAN, kart no) SELECT EDİLMEZ —
        şemada karşılıkları da yoktur, bkz. schemas.Customer.
      * `get_transactions` filtreleri WHERE'e çevrilmeli, tablo belleğe çekilmemeli.
    """

    def __init__(self, dsn: str, schema: str = "", column_map: dict | None = None) -> None:
        self.dsn = dsn
        self.schema = schema
        self.column_map = column_map or {}

    def _not_implemented(self, method: str):
        raise NotImplementedError(
            f"SqlDataSource.{method} henüz yazılmadı. Gerçek veri erişimi "
            f"açıldığında bu sınıfı doldur, sonra `python -m src.data.validate` "
            f"çalıştırıp şema uyumunu doğrula."
        )

    def get_customers(self) -> pd.DataFrame:
        self._not_implemented("get_customers")

    def get_transactions(self, customer_id=None, since=None) -> pd.DataFrame:
        self._not_implemented("get_transactions")

    def get_campaigns(self, active_only=False, as_of=None) -> pd.DataFrame:
        self._not_implemented("get_campaigns")

    def get_interactions(self) -> pd.DataFrame:
        self._not_implemented("get_interactions")


def build_data_source(config: dict) -> DataSource:
    """config.yaml'a bakarak doğru kaynağı kurar.

    Uygulamanın hiçbir yerinde `JsonDataSource(...)` doğrudan çağrılmaz;
    herkes bu fabrikayı kullanır. Kaynak değişimi tek satırlık konfig işi olur.
    """
    data_cfg = config["data"]
    kind = data_cfg.get("source", "json")

    if kind == "json":
        json_cfg = data_cfg["json"]
        return JsonDataSource(
            # resolve_path: config'teki göreli yol proje köküne göre çözülür, böylece
            # kaynak hangi dizinden çalıştırılırsa çalıştırılsın aynı veriyi bulur
            # (Streamlit ve uvicorn farklı cwd'den başlatılabiliyor).
            raw_dir=resolve_path(json_cfg["raw_dir"]),
            filenames={
                "customers": json_cfg["customers_file"],
                "transactions": json_cfg["transactions_file"],
                "campaigns": json_cfg["campaigns_file"],
                "interactions": json_cfg["interactions_file"],
            },
        )

    if kind == "sql":
        sql_cfg = data_cfg["sql"]
        return SqlDataSource(
            dsn=sql_cfg["dsn"],
            schema=sql_cfg.get("schema", ""),
            column_map=sql_cfg.get("column_map", {}),
        )

    raise ValueError(
        f"Bilinmeyen veri kaynağı: {kind!r}. config.yaml -> data.source "
        f"'json' veya 'sql' olmalı."
    )


__all__ = [
    "DataSource",
    "JsonDataSource",
    "SqlDataSource",
    "build_data_source",
    "TABLE_SCHEMAS",
]
