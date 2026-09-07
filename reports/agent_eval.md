# Ajan Değerlendirme Raporu

> ℹ️ Bu koşu **deterministik yedek yolu** ölçer (LLM yok). Şablon metinler kendi denetimlerinden geçtiği için %100 temiz çıkması beklenir; değeri budur: yedek yolun bozulduğunu gösteren bir taban ölçümü.

- Çalışma: `2026-08-20T10:53:24.496843+00:00`
- Sağlayıcı: **mock**
- Örnek: 4 müşteri × 4 istek = **16 koşu** (seed 42)

## Cevap durumu

| Durum | Adet | Oran |
|---|---:|---:|
| temiz | 16 | %100 |
| duzeltildi | 0 | %0 |
| yedek | 0 | %0 |
| hata | 0 | %0 |

**Müşteriye giden doğrulanmış metin oranı: %100** (denetimden geçen + düzelttirilip geçen). Kalanına şablon metin gitti — yani hiçbir koşuda doğrulanmamış rakam müşteriye ulaşmadı.

## Denetimin yakaladıkları

Bu koşuda ihlal yakalanmadı.

## Araç kullanımı

| Araç | Çağrı |
|---|---:|
| `analyze_spending` | 16 |
| `get_campaign_details` | 6 |
| `score_campaigns` | 4 |

## Süre ve tur

- Ortalama 0.18 sn · medyan 0.03 sn · en yavaş 2.29 sn
- Ortalama araç turu: 2.12

## Sert değişmez

✅ İzin vermeyen müşterinin cevabında kampanya kodu geçmedi (1 öneri koşusu, 4 koşu toplam).
