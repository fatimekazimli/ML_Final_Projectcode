# Decision Tree hissəsi — Ilaha Gadirli

Bu hissə `src/decision_tree.py` içində NumPy ilə yazılmış klassifikasiya və reqressiya ağacını, `tests/test_tree.py` yoxlamalarını və `src/tree_experiments.py` real-data təcrübəsini əhatə edir. Ağac modulu scikit-learn modelindən istifadə etmir; scikit-learn yalnız testdə və müqayisə skriptində işlədilir.

## Təkrar işlətmək

Python 3.11/3.12 və repodakı `requirements.txt` ilə mühit qurduqdan sonra Kaggle [Binaaz sale project](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project) version 1 arxivinin `house_sale.csv` faylını `data/bina_az_sale.csv` adı ilə qoyun. Data 16 oktyabr 2026-da endirilib; xam CSV repoya daxil edilmir.

```bash
python -m pytest -q tests/test_tree.py
python -m src.tree_experiments --train-cap 5000
```

İkinci komanda `results/tree_*` fayllarını və `report/figures/tree_*` qrafiklərini yaradır. Deterministik split, train altseçməsi və baseline üçün seed 42-dir. Vaxt ölçüləri maşına görə dəyişir. Təcrübə `data_prep.py`-ın hələ inteqrasiya olunmamış vəziyyətindən asılı olmamaq üçün öz dar, train-only preprocessing yolundan istifadə edir; komanda yekun `run_all.py`-də eyni təmizləmə/split qaydasını bütün modellərə tətbiq etməlidir.

## Müdafiə üçün əsas fikir

Hər düyündə hər feature sıralanır, fərqli qonşu dəyərlərin orta nöqtəsi namizəd threshold olur. Gini və entropy sinif qarışıqlığını, MSE isə hədəfin dispersiyasını ölçür; ən böyük impurity azalmasını verən etibarlı split seçilir. Kümülyativ cəmlər bütün threshold-ları ayrıca massiv kəsmədən hesablamağa imkan verir. `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` böyüməni məhdudlaşdırır. Leaf klassifikasiyada sinif nisbətlərini, reqressiyada ortanı qaytarır. Feature importance hər split-in train sətrinə görə çəkili impurity azalmasından gəlir.

Real datada 100,775 xam sətir var; URL üzrə son scrape saxlanandan və sabit sərhədlərlə filtrdən sonra 62,703 qalır. Bölünmə 43,891/9,406/9,406-dır. Train medianı 212,000 AZN, feature sayı 58-dir. Öz və sklearn ağacı eyni 5,000 train sətrində öyrədilib; validation və test tam splitlərdə qiymətləndirilib. Test RMSE 389,483 AZN, F1 0.8853 və AUC 0.9513-dür. Seçim yalnız validation ilə edilib: reqressiya üçün MSE/dərinlik 12/leaf 5, təsnifat üçün entropy/dərinlik 8/leaf 20.

Nəticənin zəif tərəfləri: yüksək qiymətli evlərdə və torpaq elanlarında AZN xətası böyükdür; tier sərhədinə yaxın elanların 42.1%-i səhv təsnif edilir. Bu rəqəmlər qrup ölçüsü və qiymət fərqlərindən də təsirlənir. Feature importance korrelyasiya və çoxsaylı split namizədlərinə həssasdır, səbəb-nəticə kimi yozulmamalıdır.

Hesabata əlavə üçün `report/tree_section.tex` hazırdır; ayrıca yığılan `report/tree_preview.pdf` yalnız həmin bölmənin yoxlama nüsxəsidir. Təqdimatın 3-cü və 6-cı slaydları `presentation/tree_slides.tex` və iki səhifəlik `presentation/tree_slides.pdf`-dədir; Şəxs 3 onları yekun təqdimata daxil edə bilər. Şəxs 3-ün bütün modellər üzrə CV mean ± std cədvəli və Şəxs 1-in yekun `run_all.py` inteqrasiyası hələ komanda tərəfindən tamamlanmalıdır. AI yardımı barədə açıqlama hesabat fragmentindədir.
