# SGBlur-Video

**Floutage des vidéos de terrain pour [Panoramax](https://panoramax.fr), au service de la vie privée.**
SGBlur-Video est l'équivalent vidéo de [SGBlur](https://gitlab.com/panoramax/server/sgblur) :
il repère les visages, les plaques d'immatriculation et les panneaux de
signalisation dans les vidéos (dashcam, vélo, piéton, 360°), **floute de façon
irréversible les visages et les plaques sur toutes les images où ils
apparaissent**, et renvoie **une annotation Panoramax par panneau physique**.

🇬🇧 [Read in English](README.md)

> **État : pré-alpha.** La conception est terminée et le squelette du projet est
> en place ; le traitement vidéo est en cours d'écriture. Rien n'est encore
> utilisable en production.

## Pourquoi la vidéo demande plus qu'un floutage image par image

Un visage flouté sur 299 images et net sur une seule est un échec. SGBlur-Video :

- détecte sur **chaque** image, à plusieurs échelles (et par tuiles pour la 360° en 8K) ;
- **suit** les objets dans le temps et floute aussi les images où le détecteur les a ratés ;
- floute quelques images **avant et après** chaque piste, avec une marge autour de chaque boîte ;
- floute aussi les **détections isolées de faible confiance** : mieux vaut trop flouter que pas assez ;
- utilise un flou **irréversible** (mosaïque + flou, ou aplat) ;
- **ne floute jamais les panneaux** : ils sont dédoublonnés et renvoyés comme annotations sémantiques (`osm|traffic_sign=yes`, mêmes tags que SGBlur) ;
- ne conserve aucune vidéo originale après le traitement, même en cas d'erreur.

## Démarrage (développement)

Prérequis : Python 3.14, [uv](https://docs.astral.sh/uv/), macOS (Apple Silicon) ou Linux.

```bash
git clone https://github.com/KylianAUBRY/SGBlur_Video.git
cd SGBlur_Video
uv sync
uv run sgblur-video --help
```

Le pipeline en ligne de commande arrive à l'étape 4, l'API HTTP et
`docker compose up` à l'étape 6. La feuille de route complète est dans le
[README anglais](README.md#roadmap).

## Documentation

La documentation (en anglais) est dans [`docs/`](docs/index.md) : architecture,
pipeline, contrat d'API, configuration, décisions (ADR).

## Licence

Le code de ce dépôt est sous [licence MIT](LICENSE). Il s'appuie à l'exécution
sur [Ultralytics](https://github.com/ultralytics/ultralytics), sous licence
AGPL-3.0 : déployer le service avec cette dépendance est soumis aux conditions
de cette licence. Voir [docs/license.md](docs/license.md) et
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Contribuer

Les contributions sont bienvenues : lisez [CONTRIBUTING.md](CONTRIBUTING.md) (en
anglais) et le [code de conduite](CODE_OF_CONDUCT.md). Une **fuite de vie
privée** (visage ou plaque resté visible) se signale avec le modèle de ticket
dédié, **sans aucune image ni détail permettant d'identifier qui que ce soit** :
voir [SECURITY.md](SECURITY.md). Le signalement privé est prévu pour la v2
(la v1 est développée pendant un hackathon, en temps limité).
