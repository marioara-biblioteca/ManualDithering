Logica scriptului este urmatoarea: 
- am importat imaginea;
- am adus-o la o dimensiune de matrice patratica (height-ul si width-ul) puterere a lui 2 (cea mai mica apropiata de min(height, width) astfel incat sa pastram claritatea imaginii si sa putem aplica matricea de dithering pixel la pixel)
- am aplicat algoritmul de cuantizarea kmeans pentru a reduce numarul de culori la 12 (asa cum este mentionat in cerinta) - fiecare culoare din imaginea rezultata va fi "repartizata" unei culori principale (centroid) determitata cu algorimul kmeans
- am adus imaginea in tonuri de gri pentru a o reduce la dimensiunea unei matrice 2D 
- am calculat matricea de dithering pentru algoritmul de dithering ordonat dupa formula din curs
- am aplicat algoritmul de dithering pe imaginea i care am redus numarul de culori (comparam fiecare pixel din imaginea noastra cu valoarea corespunzatoare din matricea de dithering si daca este mai mic il setam pe 0 daca este mai mare il setam pe negru => o imagine comprimata doar in nuante de alb si negru)
- am afisat imaginea in toate etapele prezentate
Aici este link-ul pentru folderul din google drive cu imaginea originala (1536 x 2048 pixeli)  : https://drive.google.com/drive/u/0/folders/1ZBmIoJ7RlHfxWcGCzVj_64YTR70C7HUT