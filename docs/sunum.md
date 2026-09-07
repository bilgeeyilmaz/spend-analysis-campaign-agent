# Sunum ve Demo Senaryosu

15 dakikalık anlatım + 5 dakika demo varsayımıyla. Anlatının omurgası şu:
**problem → veri → model → ajan → güvence → sonuç.**

## Anlatı

### 1. Problem (1 dk)
Bankalar kampanyayı kaba segment kuralıyla dağıtır: "tüm kredi kartı müşterilerine
market kampanyası." Sonuç: boşa giden bütçe, ilgisiz bildirimle yorulan müşteri.

### 2. Neden ajan, neden sadece model değil (2 dk)
Ürün iki yetenek: **harcamayı anlatmak** ve **kampanya önermek**. Teklif analizin
üstüne oturur — "şunu harcadınız, **bu yüzden** şu kampanya". Sıra tersine dönerse
proje sıradan bir öneri motoruna iner.

### 3. Veri ve kaçınılan tuzak (3 dk) — *anlatının en güçlü yeri*
Etiket basit bir kuralla üretilseydi model o kuralı ezberler, **AUC ~0.99** çıkardı.
Bu bir başarı değil, ölçüm hatasıdır. Etiket gizli bir müşteri eğilimi + kategori
uyumu + kampanya cazibesi + gürültü ile üretiliyor; gizli terim özellik tablosunda
**yok**. Hedef bant 0.72–0.82, ölçülen gözlemlenebilir tavan **0.784**.

### 4. Model (3 dk)
- **AUC 0.778**, PR-AUC 0.553, precision@1 0.502, rastgeleye karşı **1.85×**
- Tek satırlık kategori kuralı PR-AUC'de modeli yeniyor (0.567) — tabloda duruyor
- **Sızıntı**: bir müşterinin ~6 teklifi aynı bloğu taşır; satır bazlı bölmede model
  kişiyi ezberler. Derin ağaçta fark **+0.09**, sığ ağaçta **~0**. Sızıntı bir bölme
  hatası kadar bir **kapasite** meselesiydi.
- Segmentasyon: k=5 en yüksek silhouette'i aldı ama en küçük kümesi nüfusun %1,1'i →
  **elendi**. Segment bir kampanya hedef kitlesidir.

### 5. Ajan ve güvenceler (3 dk)
- Uygunluk kararını LLM vermez; `check_eligibility` verir ve `opt_in_marketing`
  **ilk** kuraldır.
- PII sızıntısı önlenmiş değil, **imkânsız**: şemada alan yok.
- Dört guardrail; metindeki her sayı bir araç çıktısına dayanmalı.
- **LLM olmadan da çalışır** — anahtar yoksa deterministik yedek devreye girer.

### 6. Sonuç ve ölçüm (2 dk)
`agent_eval` ile 4 müşteri × 4 istek (gpt-oss-20b): **12 temiz, 3 düzeltildi, 1 yedek**
→ müşteriye giden doğrulanmış metin **%94**. Denetim 5 kez izlenemeyen sayı yakaladı;
**hiçbiri müşteriye ulaşmadı.** İki ayrı koşuda da aynı %94 çıktı.

### 7. Gerçek veriye geçiş (1 dk)
`SqlDataSource` + `column_map` + `source: sql`. Özellik, model, ajan, API'ye
dokunulmaz.

## Demo senaryosu (5 dk)

Öncesinde: `./scripts/demo_ui.sh` çalışır durumda, tarayıcı açık, **C000015** seçili.

| # | Yapılan | Söylenen |
|---|---|---|
| 1 | Harcamalarım sekmesi | "Tek rakam: son 3 ayda ne harcamış. Altında nereye — kategori satırları." |
| 2 | Öne çıkanlar satırlarını göster | "Bunları kimse elle yazmadı; 3 düzenli ödeme ve 5 olağandışı hareket tespit edildi." |
| 3 | Ajana sor → hazır soru tıkla | "Soru serbest; ajan hangi aracı çağıracağına kendi karar veriyor." |
| 4 | Cevabın altındaki araç izini göster | "Cevabın gerekçesi burada: hangi araç, hangi argümanla." |
| 5 | 🎯 Kampanya öner | "Önce analiz, sonra teklif. Ve teklif modelin skoruna değil, önce iş kurallarına takılıyor." |
| 6 | Denetim rozetini göster | "Metindeki her sayı bir araç çıktısına dayanıyor — bunu makine kontrol etti." |
| 7 | Müşteriyi **C000001** yap | "Bu müşterinin pazarlama izni yok. Model skoru ne olursa olsun kampanya çıkmayacak." |

**Süre uyarısı:** 5. adım (kampanya önerisi) canlı ölçümde **~35 sn** sürüyor —
gpt-oss-20b yavaş. O boşlukta susma: araç izini aç ve "şu anda üç aracı sırayla
çağırıyor" diye anlat. İstersen sunumdan hemen önce aynı müşteride bir kez çalıştır,
ikinci çağrı önbellekten hızlı döner.

**Yedek plan:** arayüz açılmazsa `/docs` üzerinden `POST /chat`; ağ yoksa sistem
deterministik yedeğe düşer ve demo yine tamamlanır (bunu **anlatarak** göster, kusur
gibi değil tasarım gibi).

## Beklenen sorular

| Soru | Cevap |
|---|---|
| "AUC 0.78 düşük değil mi?" | Gözlemlenebilir tavan 0.784. Daha yükseği ancak sızıntıyla çıkar; ölçtük, +0.003. |
| "Zaten segmentimiz var." | O ticari segment (bakiye/gelir). Bu davranışsal: iki affluent müşteriden biri seyahat ediyor, diğeri market alışverişi yapıyor. |
| "LLM yanlış şey söylerse?" | Dört denetim + bir düzeltme hakkı + şablon yedek. Ölçüldü: 3 uydurma yakalandı, 0'ı müşteriye ulaştı. |
| "KVKK?" | Şemada PII alanı yok; `opt_in_marketing` ilk kural; `gender` modele girmiyor; her cevap audit log'da. |
| "Kural bazlı yapsak olmaz mıydı?" | Kural güçlü bir baseline ve PR-AUC'de kazanıyor — tabloda duruyor. Model AUC ve precision@1'de önde; asıl fark yeni kampanyada: kural elle yazılır, model yeniden eğitilmez. |
| "Gerçek veride çalışır mı?" | Göç yüzeyi dört sorgu + bir kolon eşlemesi. Şema sözleşmesi bunun için var. |

## Rakamlar (ezberlenecek)

2.000 müşteri · 362k işlem · 7.857 teklif · 87 özellik · **AUC 0.778** · 1.85× ·
tavan 0.784 · 3 davranışsal segment · **275 test** · doğrulanmış metin **%94** · demo akışı 38 sn
