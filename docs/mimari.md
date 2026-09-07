# Mimari

Sistem beş katmandan oluşur. Her katman yalnız bir alttakinin **sözleşmesine**
bağlıdır, uygulamasına değil: veri kaynağı JSON'dan SQL'e geçtiğinde özellik,
model, ajan ve API kodu değişmez.

```mermaid
flowchart LR
    DS["Veri Kaynağı<br/><small>JSON | SQL</small>"] --> F["Özellikler<br/><small>RFM + kategori + trend</small>"]
    F --> M["Modeller<br/><small>KMeans + LightGBM</small>"]
    M --> A["Ajan<br/><small>araç çağırma + guardrails</small>"]
    A --> API["FastAPI"]
    API --> UI["Streamlit"]
    DS -. "sözleşme" .-> SC["schemas.py"]
    SC -. "doğrulama" .-> V["validate.py"]
```

`src/data/schemas.py` sözleşmedir. `TABLE_SCHEMAS` ve `required_columns()`,
`source.py` ve `validate.py` tarafından zorlanır; alan eklemek Pydantic modelini
değiştirmek demektir, yalnız üreteci değil.

## Ajanın bir soruyu cevaplaması

```mermaid
sequenceDiagram
    participant K as Kullanıcı
    participant API as FastAPI
    participant O as Orkestratör
    participant L as LLM (Groq)
    participant T as Araçlar
    participant G as Guardrails

    K->>API: POST /chat {customer_id, question}
    API->>O: Agent.chat()
    loop en çok max_tool_iterations
        O->>L: mesajlar + TOOL_SCHEMAS
        L-->>O: araç çağrısı
        O->>T: analyze_spending / score_campaigns / check_eligibility
        T-->>O: sonuç
        Note over O: mask_payload() — PII ayıklanır, yük kırpılır
        O->>L: araç sonucu (maskeli)
    end
    L-->>O: metin
    O->>G: denetle(metin, araç kayıtları)
    alt geçti
        G-->>O: temiz
    else ihlal var
        G-->>O: ihlaller
        O->>L: düzeltme talebi (bir hak)
        Note over O: yine geçmezse deterministik şablon metin
    end
    O->>API: AgentCevabi
    Note over O: her cevap logs/audit.jsonl'a yazılır
    API-->>K: answer + tool_calls + uyarilar
```

Üç şey bu diyagramda görünmeli:

- **Uygunluk kararını LLM vermez.** `check_eligibility` bir araçtır; çıktısı
  bağlayıcıdır ve `score_campaigns` skorlamadan önce kendisi eler. "LLM kontrolü
  atladı" ulaşılabilir bir durum değildir.
- **Maskeleme LLM'e gitmeden önce olur** ve kayda maskeli hâli girer. Denetçi ve
  audit log, modelin gerçekten gördüğü veriyi görür.
- **LLM olmadan da çalışır.** `GROQ_API_KEY` yoksa `MockProvider` devreye girer;
  bu bir test taklidi değil, üretim yedeğidir.

## Guardrails

Metin müşteriye gitmeden dört denetimden geçer:

| Denetim | Yakaladığı | Neden ayrı |
|---|---|---|
| `izlenebilirlik` | metindeki her sayı bir araç çıktısına dayanmalı | uydurma rakam |
| `odul` | ödül oranı önerilen kampanyanınkiyle eşleşmeli | "%25 nakit iade" izlenebilirlikten kaçar: 25 veride var ama başka bir şeyin payı |
| `tavsiye` | yatırım/tasarruf tavsiyesi | düzenlemeye tabi faaliyet |
| `kampanya` | katalog dışı kampanya kodu | uydurma teklif |

Denetimden geçemeyen metin için bir düzeltme denenir; o da tutmazsa deterministik
şablon metne düşülür. **`uyarilar` dolu olması cevabın güvensiz olduğu anlamına
gelmez** — ayrımı `fallback` verir.

## Katmanların kuralları

| Katman | Kural |
|---|---|
| `data/` | PII alanı şemada yok; sızıntı önlenmiş değil, **imkânsız** |
| `features/` | `gender`, `city` tabloya hiç girmez; `opt_in_marketing` tabloda durur ama model girdisi değildir |
| `models/` | tek propensity modeli tüm kampanyalar için; `campaign_id` özellik olamaz |
| `agent/` | analiz kampanya kataloğunu okumaz; öneri analizin üstüne oturur |
| `api/` | ince katman; cevap gövdesi `AgentCevabi.to_dict()`'in aynısı |
| `ui/` | servisin HTTP istemcisi; `src` içinden hiçbir şey import etmez |

## Gerçek veriye geçiş

```mermaid
flowchart LR
    subgraph Değişen
        S["SqlDataSource<br/><small>4 sorgu</small>"]
        C["config.yaml<br/><small>column_map, source: sql</small>"]
    end
    subgraph Değişmeyen
        F["features/"] --> M["models/"] --> A["agent/"] --> API["api/"] --> UI["ui/"]
    end
    S --> F
```

1. `src/data/source.py` içindeki `SqlDataSource`'u doldur (dört sorgu).
2. `config.yaml → data.sql.column_map` ile banka kolonlarını şema alanlarına eşle.
3. `config.yaml → data.source: sql`.
4. `python -m src.data.validate` — PII kolonu görürse doğrulama hatayla durur.

Göç yüzeyi bu kadardır. Özellik üretimi, modeller, ajan ve API'ye dokunulmaz.
