# Autoencodeur texte <-> image : resultats

Critere de reussite pre-enregistre : >= 99 % des caracteres relus correctement (validation).

| bras | morceau (car.) | params | car. corrects | morceaux exacts | perte (nats/car) | continuite | reussi |
|:--|--:|--:|--:|--:|--:|--:|:-:|
| cumul | 8 | 674233 | 100.0 % | 99.6 % | 0.001 | 0.68 | oui |
| cumul | 16 | 872889 | 99.9 % | 98.1 % | 0.004 | 0.71 | oui |
| cumul | 32 | 1270201 | 97.9 % | 60.9 % | 0.065 | 0.61 | non |
| cumul | 64 | 2064825 | 53.0 % | 0.0 % | 1.213 | 0.27 | non |
| image | 8 | 1238574 | 100.0 % | 99.9 % | 0.000 | - | oui |
| image | 16 | 1634862 | 100.0 % | 100.0 % | 0.000 | - | oui |
| image | 32 | 2427438 | 100.0 % | 98.5 % | 0.002 | - | oui |
| image | 64 | 4012590 | 100.0 % | 98.3 % | 0.001 | - | oui |
| vector | 8 | 1238496 | 100.0 % | 100.0 % | 0.000 | - | oui |
| vector | 16 | 1634784 | 100.0 % | 99.8 % | 0.000 | - | oui |
| vector | 32 | 2427360 | 100.0 % | 99.0 % | 0.001 | - | oui |
| vector | 64 | 4012512 | 100.0 % | 98.5 % | 0.001 | - | oui |
| none | 8 | 1216552 | 100.0 % | 100.0 % | 0.000 | - | oui |
| none | 16 | 1612840 | 100.0 % | 99.9 % | 0.000 | - | oui |
| none | 32 | 2405416 | 100.0 % | 99.8 % | 0.000 | - | oui |
| none | 64 | 3990568 | 100.0 % | 98.1 % | 0.001 | - | oui |

Continuite (bras cumul) : pas entre frames consecutives / distance entre frames de textes differents ; < 1 = video continue.

- cumul L=8, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- cumul L=16, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- cumul L=32, precision par position (debut / milieu / fin du morceau) : 99 % / 98 % / 98 %
- cumul L=64, precision par position (debut / milieu / fin du morceau) : 79 % / 47 % / 86 %
- image L=8, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- image L=16, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- image L=32, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- image L=64, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- vector L=8, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- vector L=16, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- vector L=32, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- vector L=64, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- none L=8, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- none L=16, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- none L=32, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %
- none L=64, precision par position (debut / milieu / fin du morceau) : 100 % / 100 % / 100 %

Plus grand morceau reussi par bras : cumul = 16, image = 64, none = 64, vector = 64
