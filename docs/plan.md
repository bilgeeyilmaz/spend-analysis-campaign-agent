# 15 İş Günlük Proje Planı

Başlangıç: 10 Ağustos 2026 (Pazartesi) · Bitiş: 28 Ağustos 2026 (Cuma)

```mermaid
gantt
    title Kampanya Öneri Agent Sistemi - 15 İş Günü
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    excludes weekends

    section Hafta 1 - Veri ve Model
    Gün 1  İskelet, veri şeması, kampanya kataloğu   :done,   d1,  2026-08-10, 1d
    Gün 2  Sentetik veri üreteci                     :done,   d2,  after d1,   1d
    Gün 3  EDA ve özellik üretimi (RFM, kategori)    :active, d3,  after d2,   1d
    Gün 4  Segmentasyon (KMeans) ve persona yorumu   :        d4,  after d3,   1d
    Gün 5  Propensity modeli ve baseline karşılaştırma :      d5,  after d4,   1d
    Model hattı uçtan uca çalışıyor                  :milestone, m1, after d5, 0d

    section Hafta 2 - Agent ve Servis
    Gün 6  Araç katmanı - harcama analizi + skorlama  :       d6,  after d5,   1d
    Gün 7  Groq entegrasyonu ve tool-calling döngüsü  :       d7,  after d6,   1d
    Gün 8  Guardrails, PII maskeleme, audit log       :       d8,  after d7,   1d
    Gün 9  FastAPI servisi                            :       d9,  after d8,   1d
    Gün 10 Uçtan uca MVP toparlama                    :crit,  d10, after d9,   1d
    MVP SERT KONTROL NOKTASI                          :milestone, crit, m2, after d10, 0d

    section Hafta 3 - Demo, Kalite, Teslim
    Gün 11 Streamlit demo arayüzü                     :       d11, after d10,  1d
    Gün 12 Test suite ve offline değerlendirme        :       d12, after d11,  1d
    Gün 13 Dokümantasyon ve mimari diyagram           :       d13, after d12,  1d
    Gün 14 Sunum hazırlığı ve demo senaryosu          :       d14, after d13,  1d
    Gün 15 Prova ve teslim                            :crit,  d15, after d14,  1d
    TESLİM                                            :milestone, crit, m3, after d15, 0d
```

## Kritik yol

Kırmızı işaretli kalemler kaydırılamaz. **Gün 10** projenin tek sert kontrol
noktasıdır: o gün uçtan uca `curl` ile çalışan bir MVP yoksa kapsam daraltılır.

Feda sırası (gerekirse, bu sırayla):

1. Streamlit arayüzü — demo `/docs` üzerinden yapılır
2. Offline değerlendirme raporu
3. Segmentasyon görselleri

**Model ve agent feda edilmez** — projenin anlatısı bu ikisidir.
