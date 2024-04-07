## Temat

Zintegrowane uwzględnianie kosztów pomyłek przy tworzeniu modeli klasyfikacji przez próbkowanie przykładów klas o niskich kosztach pomyłek, replikację przykładów klas o wysokich kosztach pomyłek oraz zmianę etykiet (metoda MetaCost). Implementacja w formie opakowania umożliwiającego użycie dowolnego algorytmu klasyfikacji dostępnego w środowisku R lub Python stosującego standardowy interfejs wywołania. Funkcje do tworzenia modelu i predykcji. Badanie wpływu uwzględniania kosztów pomyłek na jakość modeli klasyfikacji tworzonych za pomocą wybranych algorytmów dostępnych w środowisku R lub Python.

## Wprowadzenie

Większość algorytmów uczących modele klasyfikujące zakłada, że wszystkie możliwe rodzaje pomyłek mają taki sam koszt. W rzeczywistych zastosowaniach często nie jest to prawdą, pomyłka jednego rodzaju jest o wiele bardziej szkodliwa niż innego rodzaju. Modyfikowanie istniejących już algorytmów, tak aby uwzględniały te kwestie wymaga osobnego podejścia do każdego z nich, więc zajęłoby dużo czasu i może nie być trywialne.

W artykule *MetaCost: A General Method for Making Classifiers Cost-Sensitive* Pedro Domingos proponuje meta algorytm, który sprawia, że dowolny model klasyfikujący jest w stanie uwzględniać różne koszty poszczególnych pomyłek. Jest to możliwe poprzez potraktowanie modelu jako czarnej skrzynki. Poprzez modyfikacje zbioru treningowego i obserwacje zmian w predykcjach modelu, można odpowiednio nauczyć model, tak aby kosztu pomyłek były uwzględnione.

## Implementacja

Do realizacji projektu, z dwóch możliwości, wybrany został język Python. Wybrane algorytmy będą pochodziły z modułu scikit-learn (sklearn).
Standardowym sposobem wywołania uczenia modelu w tym module jest funkcja *fit(X, Y)*, a do predykcji *predict(Y)*. To pozwala na zaimplementowanie algorytmu MetaCost zgodnie z jego przeznaczeniem, tj. traktując algorytm wewnętrzny jak czarną skrzynkę. Przekazując do algorytmu obiekt klasy jednego z klasyfikatorów, można wywołać te funkcje bez wiedzy jaki algorytm jest używany pod spodem.

## Lista algorytmów

W celu przeprowadzenia eksperymentów zostaną wykorzystane następujące algorytmy:

- drzewo decyzyjne `sklearn.tree.DecisionTreeClassifier`
- las losowy `sklearn.ensemble.RandomForestClassifier`
- naiwny klasyfikator bayesowski `sklearn.naive_bayes.GaussianNB`
- sieć neuronowa `sklearn.neural_network.MLPClassifier`

Wybór algorytmów nie jest przypadkowy. Starano się wybrać algorytmy o różnych zasadach działania.

## Plan badań

### Cel poszczególnych eksperymentów

Algorytm MetaCost pozwala obsługiwać przykłady, które są trudne do klasyfikacji. Trudność może wynikać między innymi z:

- podobieństwa przykładów treningowych z różnych klas.
- za małej ilości przykładów treningowych — niezbilansowania zbioru danych.

Ponieważ trudno jest ocenić podobieństwo przykładów treningowych oraz problem niezbilansowanych zbiorów danych jest bardzo powszechny w realnych zastosowaniach, eksperymenty będą skupiały się na zbiorach niezbilansowanych. W tym celu postawiono następujące hipotezy:

- MetaCost powinien poprawiać wyniki dla zbiorów niezbilansowanych
- MetaCost powinien w minimalnym stopniu wpływać na wyniki dla zbiorów zbilansowanych
- MetaCost powinien działać z dowolną metodą klasyfikacji

### Charakterystyka zbiorów danych

Aby udowodnić postawione hipotezy, należy wybrać zbiory danych zbilansowane oraz niezbilansowane. Powinno się skorzystać z dużych zbiorów danych ze względu na użycie lasu losowego. Przygotowanie danych powinno zawierać eksplorację danych mającą na celu wykrycie niepoprawnych instancji danych. Następnie tak wykryte przypadki należy odpowiednio przekształcić bądź usunąć. Należy pamiętać, że chcemy zbadać scenariusz, w którym rozkład przykładów treningowych jest niejednostajny. Pytanie, jak powinno się testować taki scenariusz. Rozważamy dwie opcje — sztuczne zbilansowanie zbioru danych lub skorzystanie z metryk, które uwzględniają niezbilansowanie.

### Parametry algorytmów, których wpływ na wyniki będzie badany

Algorytm MetaCost ma szereg parametrów, których wpływ na jakość końcowego modelu można sprawdzić. Dla ustalonego wewnętrznego algorytmu, zbioru trenującego, macierzy kosztów pomyłek, do określenia pozostają:

- liczba prób, które zostaną wygenerowane,
- liczba próbek w każdej próbie,
- czy do ewaluacji modeli używane będą próbki wszystkie, czy tylko te nie występujące w jego próbie.

Wpływ tych trzech parametrów będzie zbadany, alby lepiej zrozumieć działanie algorytmu zewnętrznego.

### Miary jakości i procedury oceny modeli

Tak jak wcześniej wspomniano jedną z technik oceny, będzie wykorzystanie metryk, które uwzględniają niezbilansowanie zbiorów. W tym celu uśrednianie metryk będzie się odbywać, uwzględniając ilość instancji. Metryki, które będą stosowane:

- precyzja
- odzysk
- miara F1
- macierz pomyłek
- dokładność

przykładowy wynik może wyglądać następująco:

```sh
                  metric1    metric2    ...

     class 1       1.00      0.67      0.80
     class 2       0.00      0.00      0.00
     class 3       0.00      0.00      0.00

   micro avg       1.00      0.67      0.80
   macro avg       0.33      0.22      0.27
weighted avg       1.00      0.67      0.80
```
