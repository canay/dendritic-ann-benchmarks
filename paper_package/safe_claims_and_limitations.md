# Güvenli İddialar ve Zorunlu Sınırlılıklar

## "Yöntem sorunu" ile ne kastediliyor?

Buradaki mesele deneylerin başarısız olması değildir. Deney paketi mevcut, sonuçlar tutarlı ve ana bilimsel hikaye ayaktadır. Sorun şu tür reviewer sorularını önceden doğru yönetmektir:

- Hangi accuracy metriği ana kanıt olarak kullanıldı?
- Low-data koşullarda test seti sabit miydi?
- Seed, model kurulmadan önce de uygulanıyor muydu?

Yani problem "deneyler geçersiz" değil, "raporlama ve protokol şeffaflığı dikkatli kurulmalı" problemidir.

## Güvenle yazılabilecek iddialar

1. Çalışma, DANN ailesini parametre-eşleşmiş MLP, naive branching kontrolü ve daha yüksek kapasiteli dense baseline ile karşılaştıran kontrollü bir benchmark sunmaktadır.
2. Validation-seçimli kontrolde `DANN_LRF`, FashionMNIST full koşusunda düşük-parametreli adaylar içinde en güçlü modeldir.
3. Validation-seçimli kontrolde `DANN_RANDOM`, KMNIST full koşusunda en iyi DANN varyantıdır.
4. En iyi dendritik örnekleme stratejisi veri yapısına bağlı görünmektedir.
5. `DANN_LRF`, CIFAR-10 full koşusunda `MLP_PARAM` karşısında açık üstünlük göstermektedir.
6. `DANN_LRF`, birçok koşulda `NAIVE_BRANCH` üstüne küçük ama yön olarak tutarlı kazanımlar eklemektedir.
7. `VANN_SAME`, daha yüksek parametre bütçesi ile daha yüksek mutlak accuracy verebilmektedir.
8. CPU timing sonuçları, `DANN_LRF`'nin `MLP_PARAM`'dan daha yavaş, ancak `NAIVE_BRANCH`'e göre yalnızca sınırlı ölçüde daha maliyetli olduğunu göstermektedir.
9. History dosyalarından çıkarılan validation-seçimli özetler, ana sonuç hikayesini bozmamaktadır.

## Dikkatli, sınırlı dille yazılması gereken iddialar

1. `DANN_LRF`, naive branching'in ötesinde değer katmaktadır.
   Bunun altı çizilebilir, ancak farkların çoğu koşulda küçük olduğu da açıkça yazılmalıdır.
2. Local receptive field sampling, uzamsal olarak düzenli görsel veri için uygun olabilir.
   Bu makul bir yorumdur, ispatlanmış mekanizma değildir.
3. Random sampling, KMNIST gibi daha düzensiz desenlerde avantaj sağlayabilir.
   Bu da veri destekli bir yorumdur, biyolojik veya nedensel ispat değildir.
4. DANN yapısı moderate low-data koşulunda yardımcı olabilir.
   Bu cümle ancak mevcut low-data koşularının train-only değil reduced-dataset diagnostic koşular olduğu açıkça yazılırsa kullanılmalıdır.

## Yazılmaması gereken iddialar

1. `DANN_LRF` her koşulda en iyi modeldir.
2. DANN ailesi evrensel olarak üstündür.
3. Çalışma state-of-the-art image classification sonucu sunmaktadır.
4. CIFAR-10 performansı modern CNN, ResNet, ViT veya güçlü görsel benchmark standartlarıyla rekabetçidir.
5. DANN her anlamda daha verimlidir.
6. Mevcut low-data sonuçları tek başına train-only low-data robustness ispatıdır.
7. Çalışma biyolojik mekanizmayı ispatlamaktadır.

## Makalede mutlaka açıkça yazılması gereken sınırlılıklar

1. Modeller flattened image input ile çalışmaktadır. Convolutional backbone kullanılmamıştır.
2. CIFAR-10 absolute accuracy, modern görsel sınıflandırma modellerine kıyasla düşüktür.
3. `DANN_LRF` ile `NAIVE_BRANCH` arasındaki fark çoğu koşulda küçüktür.
4. `DANN_RANDOM`, KMNIST full koşusunda `DANN_LRF`'den daha iyi sonuç vermiştir.
5. FashionMNIST 0.1 koşusunda `DANN_LRF` en iyi model değildir.
6. Mevcut historical low-data sonuçları, eski loader davranışı nedeniyle test setini de küçülten bir protokolden gelmektedir.
   Bu yüzden bu klasörler güçlü train-only low-data kanıtı değil, yardımcı reduced-dataset diagnostic sonuçlar olarak sunulmalıdır.
7. Historical accuracy özetlerinde dışa aktarılan `best_test_acc`, validation ile seçilen tek epoch test skoru değil, epochlar boyunca görülen en yüksek test accuracy'dir.
8. Buna karşılık, history dosyaları üzerinden yapılan validation-seçimli sağlamlık kontrolü ana hikayeyi korumaktadır ve ana tablo metrikleri bunun üstünden kurulmalıdır.
9. Accuracy koşuları karışık donanımda tamamlanmıştır.
   Bu durum accuracy yorumu için kritik değildir, çünkü wall-clock kıyasları accuracy koşularından değil, ayrı CPU timing koşularından alınmaktadır.
10. Historical sonuç paketi, seed-before-model-construction düzeltmesinden önce üretilmiştir.
    Kod artık gelecekteki koşular için bu sırayı düzeltmektedir, ancak mevcut historical sonuçlar bu değişiklikten önce alınmıştır.
11. Timing sonuçları ayrı bir süreçte ve CPU-only göreli maliyet analizi olarak yorumlanmalıdır.
12. Timing klasörlerindeki yeni `timing_run_config.json` dosyaları, mevcut timing çıktıları için manuel şeffaflık kaydıdır; sonradan uydurulmuş sonuç değildir.

## Makalede önerilen dürüst konumlandırma

En iyi ana tez şu çizgidedir:

> This study does not present DANNs as universal accuracy-maximizing classifiers. Instead, it provides controlled evidence that dendritic computation can improve parameter-efficient classification under matched baselines, while the advantage over naive branching and the best sampling strategy both remain dataset-dependent.

## Pratik yayın tavsiyesi

Bu haliyle full-data ve timing omurgası makale yazımı için yeterince güçlüdür. Low-data kısmı ise şu şekilde konumlandırılmalıdır:

- şu an için yardımcı/diagnostic sonuç
- ana robustness iddiası değil
- fixed-test rerun yapılırsa güçlenecek bölüm

Ek koşu almak zorunlu değil, ama low-data bölümü ana iddiaya dönüştürülecekse `fixedtest` klasörlerine yeniden koşu alınması en temiz çözümdür.
