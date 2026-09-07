# Ajan Değerlendirme Raporu

- Çalışma: `2026-08-21T08:09:59.389094+00:00`
- Sağlayıcı: **groq** · model `openai/gpt-oss-20b`
- Örnek: 4 müşteri × 4 istek = **16 koşu** (seed 42)

## Cevap durumu

| Durum | Adet | Oran |
|---|---:|---:|
| temiz | 12 | %75 |
| duzeltildi | 3 | %19 |
| yedek | 1 | %6 |
| hata | 0 | %0 |

**Müşteriye giden doğrulanmış metin oranı: %94** (denetimden geçen + düzelttirilip geçen). Kalanına şablon metin gitti — yani hiçbir koşuda doğrulanmamış rakam müşteriye ulaşmadı.

## Denetimin yakaladıkları

| İhlal | Adet |
|---|---:|
| `izlenemeyen_sayi` | 5 |
| `arac_hatasi` | 1 |

## Araç kullanımı

| Araç | Çağrı |
|---|---:|
| `analyze_spending` | 16 |
| `score_campaigns` | 10 |
| `get_campaign_details` | 1 |

## Süre ve tur

- Ortalama 26.99 sn · medyan 25.63 sn · en yavaş 63.68 sn
- Ortalama araç turu: 3.0

## Sert değişmez

✅ İzin vermeyen müşterinin cevabında kampanya kodu geçmedi (1 öneri koşusu, 4 koşu toplam).
