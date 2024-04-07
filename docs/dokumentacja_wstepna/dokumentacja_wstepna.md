# Zaawansowane uczenie maszynowe

## Projekt

Semestr 24L

Piotr Hondra</br>
Jan Jeschke

### Temat
Zintegrowane uwzględnianie kosztów pomyłek przy tworzeniu modeli klasyfikacji przez próbkowanie przykładów klas o niskich kosztach pomyłek, replikację przykładów klas o wysokich kosztach pomyłek oraz zmianę etykiet (metoda MetaCost). Implementacja w formie opakowania umożliwiającego użycie dowolnego algorytmu klasyfikacji dostępnego w środowisku R lub Python stosującego standardowy interfejs wywołania. Funkcje do tworzenia modelu i predykcji. Badanie wpływu uwzględniania kosztów pomyłek na jakość modeli klasyfikacji tworzonych za pomocą wybranych algorytmów dostępnych w środowisku R lub Python.

## Wprowadzenie
<!-- TODO: Na czym polega algorytm i po co go stosować -->

Większość algorytmów uczących modele służące do klasyfikacji zakłada, że wszystkie możliwe rodzaje pomyłem mają taki sam koszt. W rzeczywistych zastosowaniach często nie jest to prawdą, pomyłka jednego rodzaju jest o wiele bardziej szkodliwa niż innego rodzaju. Modyfikowanie istniejących już algorytmów, tak aby uwzględniały te kwestie wymaga osobnego podejścia do każdego z nich i może nie być trywialne.

W artykule *MetaCost: A General Method for Making Classifiers Cost-Sensitive* Pedro Domingos proponuje meta algorytm, który sprawia, że dowolny model klasyfikujący jest w stanie uwzględniać różne koszty poszczególnych pomyłek. Jest to możliwe poprzez potraktowanie modelu jako czarnej skrzynki. Modyfikując zbiór treningowy jest możliwe otrzymanie różnych modeli i ocenianie ich na podstawie rezultatów.

## Implementacja
<!--
TODO:
    - Będziemy stosować sklearn.
    - W jaki sposób napiszemy algorytm żeby współpracował z sklearn?
    - Wiedząc, że dopoasowanie do danych wykorzystuje metodę fit, też zaimplementujemy fit itd
  -->

Do realizacji projektu, z dwóch możliwości, wybrany został język Python. Wybrane algorytmy będą pochodziły z modułu scikit-learn (sklearn).
Standardowym sposobem wywołania uczenia modelu w tym module jest funkcja *fit(X, Y)*, a do predykcji *predict(Y)*. To pozwala na zaimplementowanie algorytmu MetaCost zgodnie z jego przeznaczeniem, tj. traktując jak czarną skrzynkę. Przekazując do algorytmu obiekt klasy jednego z klasyfikatorów, można wywołać te funkcje bez wiedzy jaki algorytm jest używany pod spodem.

## Lista algorytmów
<!-- TODO: algorytmy związane z klasyfikacją, np.
    - drzewa decyzyjne
    - naiwny bayes?
    - svm?
    - wszystko ofc z sklearn
 -->
listę algorytmów, które będą wykorzystane w eksperymentach (ze wskazaniem wykorzystywanych bibliotek, i klas/funkcji)

## plan badań

### cel poszczególnych eksperymentów (pytania, na które będzie poszukiwana odpowiedź, lub hipotezy do weryfikacji)
<!-- TODO:
    - wypisanie hipotez np:
      - algorytm powinien lepiej działać dla zbiorów niezbalansowanych
      - algorytm powinien nie psuć wyników dla zbiorów zbalansowanych
      - algorytm powinien działać z dowolną metodą klasyfikacji
      - ....
      - na marginesie fajnie by porównać wyniki z podobnymi algo jak np. weighted sampling czy SMOTE
 -->

### charakterystykę zbiorów danych, które będą wykorzystane (oraz ewentualnych czynności związanych z przygotowaniem danych)
<!--
TODO:
    - znależć zbiory danych, którę będą dobre do klasyfikacji i będą spełniały hipotezy
    - feaure engineering:
      - normalizacje  i czyszczenie jesli bedzie potrzeba
 -->

### parametry algorytmów, których wpływ na wyniki będzie badany
<!-- TODO:
    - nie wiem tutaj o co chodzi
 -->

### miary jakości i procedury oceny modeli
<!-- TODO:
    - wszystko metryki które służą oceny klasyfikacji
    -

 -->

### otwarte kwestie wymagające późniejszego rozwiązania (wraz z wyjaśnieniem powodów, dla których ich rozwiązanie jest odłożone na później)

