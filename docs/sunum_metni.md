# Sunum Metni

~15 dakika anlatım + 5 dakika demo. Köşeli parantezler sahne yönergesi, italik
olanlar söylenecek metin. Rakamların hepsi ölçülmüştür; ezberlenmesi gerekenler
**kalın**.

---

## 1. Açılış (30 sn)

*"Herkese merhaba. Size staj süresince geliştirdiğim projeyi anlatacağım:
harcama analizi yapan ve buna dayanarak kampanya öneren bir asistan.*

*Tek cümleyle özeti şu: **önce müşterinin harcamasını anlatıyoruz, sonra ona
uygun kampanyayı gerekçesiyle sunuyoruz.** Sıralama önemli — teklif, analizin
sonucu."*

---

## 2. Problem (1,5 dk)

*"İki taraflı bir sorunla başladım.*

*Müşteri tarafında: çoğumuz ayın sonunda 'param nereye gitti?' diye soruyoruz.
Kart ekstresine bakıyorsunuz, elli satır işlem var, hangi kategoriye ne kadar
gittiğini göremiyorsunuz.*

*Banka tarafında: kampanyalar çoğunlukla kaba kurallarla dağıtılıyor. 'Tüm kredi
kartı müşterilerine market kampanyası' gibi. Sonuç, boşa harcanan kampanya
bütçesi ve ilgisiz bildirimlerle yorulan müşteri.*

*Bu iki sorunun aynı çözümü var: müşterinin ne yaptığını gerçekten anlamak."*

---

## 3. Çözüm ve demo (4 dk) [EKRANI AÇ]

[Arayüzü aç, C000015 seçili]

*"Ekranda bir müşterinin son üç ayı var. Üstte ne kadar harcadığı, altında
nereye harcadığı — kategoriler, tutarlar, geçen döneme göre değişim.*

*Şunu vurgulamak isterim: bu kategorileri kimse elle etiketlemedi. Kart
işlemlerinden çıkarıldı."*

[Öne çıkanlar bölümünü göster]

*"Burada sistemin kendi bulduğu şeyler var: 'Bu müşteri altı aydır aynı üye
işyerine ayda ortalama şu kadar ödüyor' — yani düzenli bir gideri var. Ya da
'şu kategoride olağandışı bir hareket var'."*

[Ajana sor sekmesine geç, hazır sorulardan birine tıkla]

*"Grafiklere bakmak istemeyen müşteri doğrudan sorabiliyor. Asistan yalnız bu
müşterinin kendi verisiyle cevaplıyor."*

[Cevap gelince "Gerekçe" kutusunu aç]

*"Ve cevabın altında gerekçesi var: hangi araçları çağırdı, hangi veriye baktı.
Buraya birazdan döneceğim, çünkü projenin en önemli kısmı burası."*

[🎯 Kampanya öner butonuna bas]

*"Şimdi kampanya önerisini isteyelim. Dikkat edin: önce harcamayı özetliyor,
sonra teklifi ona bağlıyor. 'Şunu harcadınız, bu yüzden şu kampanya.'"*

---

## 4. Kampanya nasıl seçiliyor (3 dk)

*"Arka planda üç adım var.*

***Birincisi, öğrenme.** Sisteme geçmişte kime hangi kampanyanın sunulduğunu ve
kabul edilip edilmediğini verdik — 7.857 teklif. Model bu veriden hangi harcama
deseninin hangi kampanyayla eşleştiğini kendi çıkardı. Biz kural yazmadık.*

***İkincisi, puanlama.** Her müşteri-kampanya çifti için bir kabul puanı
hesaplanıyor.*

***Üçüncüsü — ve en önemlisi — kurallar.** Puan ne kadar yüksek olursa olsun,
kampanya önce iş kurallarından geçiyor: pazarlama izni var mı, bütçe kalmış mı,
tarih geçerli mi, yaş ve ürün şartları tutuyor mu. Kurala takılan kampanya
müşteriye **hiç ulaşmıyor**.*

*Yani model karar vermiyor, sadece sıralıyor.*

*Sonuç: rastgele kampanya seçiminde 100 müşteriden 27'si ilk teklifi kabul
ediyor. Modelle bu oran **%50**'ye çıkıyor."*

[Sorulursa: AUC 0,778, ölçülen tavan 0,784]

---

## 5. Müşteri grupları (1,5 dk) [segmentler.png]

*"Bir de müşterileri harcama davranışına göre gruplayan bir model var. Bankanın
mevcut segmentinden farkı şu: o bakiyeye bakıyor, bu paranın nereye gittiğine.*

*Üç grup çıktı. Soldaki grup harcamasının dörtte birini markete ayırıyor.
Ortadaki online alışveriş ve restoran ağırlıklı. Sağdaki seyahat ediyor ve
ortalama dört kat daha fazla harcıyor.*

*Bu grupların isimlerini de ben yazmadım — sistem her grubun hangi kategoride
ortalamanın üstünde olduğuna bakıp kendi adlandırdı.*

*Buradaki mesaj basit: aynı kampanyayı bu üç gruba birden göndermek bütçeyi
boşa harcamak demek."*

---

## 6. Neden güvenilir (3 dk) — **sunumun tepe noktası**

*"Şimdi en çok üzerinde durduğum konuya geleyim. Bu sistemde bir dil modeli var
ve dil modelleri bazen olmayan şeyler uydurur. Bir bankada bu kabul edilemez.*

*Kurduğumuz kural şu: **asistanın yazdığı her rakam, sistemin kendi hesabına
dayanmak zorunda.** Metin müşteriye gitmeden önce otomatik bir denetimden
geçiyor; metindeki her sayı araç çıktılarıyla karşılaştırılıyor. Dayanmayan bir
rakam varsa o cevap müşteriye gitmiyor — sistem önce modelden düzeltmesini
istiyor, olmazsa hazır şablon metne düşüyor.*

*Bunu ölçtük: bir değerlendirme koşusunda denetim **5 hatalı rakam yakaladı,
hiçbiri müşteriye ulaşmadı**.*

[Demo ekranındaki denetim rozetini göster]

*İki güvence daha var.*

*Pazarlama izni olmayan müşteriye hiçbir koşulda kampanya çıkmıyor — bu bir
model kararı değil, hukuki bir zorunluluk ve kontrol listesinde ilk sırada.*

[Müşteriyi C000001 yap]

*İşte izin vermemiş bir müşteri. Model ne derse desin, burada kampanya
görmeyeceksiniz.*

*Ve sistemde müşterinin adı, TC kimlik numarası, telefonu hiç tutulmuyor. Bunu
'silmiyoruz' demiyorum — veri şemasında böyle bir alan **yok**. Sızması mümkün
değil, çünkü orada değil."*

---

## 7. Veri (1,5 dk)

*"Bir açıklama borçluyum: bu projede gerçek müşteri verisi kullanılmadı. Veriyi
yapay olarak ürettim — 2.000 müşteri, 12 aylık 362 bin kart işlemi.*

*Ama burada dikkat ettiğim bir nokta var. Kabul davranışını üretirken kolay yolu
seçseydim — 'şu özellikteki müşteri şu kampanyayı kabul eder' diye basit bir
kural yazsaydım — model o kuralı ezberler ve neredeyse kusursuz görünürdü. Bu
sahte bir başarı olurdu.*

*Onun yerine kabul kararını modelin göremeyeceği gizli bir etkene bağladım.
Gerçek hayatta da müşterinin o günkü ruh hâlini bilemeyiz. Sonuçların gerçekçi
olmasının sebebi bu."*

---

## 8. Gerçek veriye geçiş (1 dk)

*"Sistemi baştan beri şu varsayımla kurdum: yapay veri geçici, gerçek veri
gelecek.*

*Katmanların hiçbiri verinin nereden geldiğini bilmiyor. Geçiş için yalnız veri
bağlantısının yazılması yeterli — dört sorgu ve bir kolon eşlemesi. Özellik
üretimi, modeller, asistan ve arayüz olduğu gibi kalıyor."*

---

## 9. Kapanış (1 dk)

*"Toparlarsam:*

*Müşteri tarafında finansal görünürlük — parasının nereye gittiğini görüyor ve
sorabiliyor.*

*Banka tarafında daha isabetli kampanya — ilk teklifte kabul oranı %27'den
%50'ye çıkıyor.*

*Ve her ikisi de denetlenebilir: sistemin söylediği her rakamın kaynağı var, her
cevap kayda geçiyor.*

*Teşekkür ederim, sorularınızı alabilirim."*

---

## Soru-cevap için hazır cevaplar

**"Neden isabet oranı daha yüksek değil?"**
*"Bu veride ulaşılabilecek en yüksek değeri ayrıca hesapladık: 0,784. Model
0,778'de, yani tavanın hemen altında. Daha yükseği ancak veri sızıntısıyla
çıkardı — bu yüzden yüksek bir rakam görsek sevinmez, hata arardık."*

**"Basit bir kural da aynı işi yapmaz mı?"**
*"Kısmen evet, ve bunu raporda gizlemedik: tek satırlık bir kural bir metrikte
modeli geçiyor. Ama o kural elle yazılmış. Katalog büyüdükçe her kampanya türü
için yeni kural yazmak gerekir. Model kampanyanın özelliklerinden öğreniyor."*

**"Model müşteriyi ezberlemiş olabilir mi?"**
*"Bir müşterinin ortalama altı teklifi var. Veriyi rastgele bölseydik aynı kişi
hem eğitimde hem testte olurdu. Müşteri bazlı böldük ve iki yöntemin farkını
ölçtük: 0,003. Ezber olsaydı bu fark büyük çıkardı."*

**"Yapay zekâ yanlış bir şey söylerse?"**
*"Söyleyemiyor — daha doğrusu söylediği müşteriye ulaşmıyor. Her cevap
denetimden geçiyor ve geçemeyen metin şablona düşüyor. Ölçtük: 5 hatalı rakam
yakalandı, sıfırı müşteriye ulaştı."*

**"Yeni kampanya eklenince ne olacak?"**
*"Hiçbir şey — yeniden eğitim gerekmiyor. Model kampanyanın kimliğine değil
özelliklerine bakıyor; katalogda tanımlanan yeni kampanyayı da puanlıyor."*

**"KVKK açısından durum nedir?"**
*"Kişisel veri şemada yok, dolayısıyla sızması mümkün değil. Pazarlama izni
kontrol listesinde ilk sırada ve hiçbir model skoru onu geçemiyor. Cinsiyet
verisi saklanıyor ama hedeflemede kullanılmıyor. Her öneri kayda geçiyor."*
