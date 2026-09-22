# Verification API

Microservice FastAPI (Python) chargé du futur pipeline d'analyse des chèques
(extraction de signature, comparaison, score de similarité).

## Statut actuel

Cette étape fournit la fondation du service et le prétraitement technique des images :

- `GET /api/health` : endpoint de santé.
- `POST /api/images/analyze` : analyse technique d'une image de chèque (OpenCV).
- CORS limité aux origines locales ASP.NET.
- Aucune logique métier/OCR/signature n'est encore implémentée.

## Prérequis

- Python 3.12+

## Installation

```bash
cd services/verification-api
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Configuration

Copier `.env.example` vers `.env` et ajuster si nécessaire :

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/macOS
```

| Variable | Défaut | Description |
|---|---|---|
| `APP_NAME` | `verification-api` | Nom du service |
| `APP_VERSION` | `0.1.0` | Version du service |
| `APP_ENV` | `development` | Environnement ; `development` active les endpoints dev-only (ex. `/api/signatures/debug`) |
| `HOST` | `0.0.0.0` | Interface d'écoute |
| `PORT` | `8000` | Port d'écoute |
| `CORS_ORIGINS` | `http://localhost:5285,http://127.0.0.1:5285` | Origines autorisées (séparées par des virgules) |
| `SIGNATURE_ROI_X_START` | `0.40` | Zone candidate de signature — début X relatif (0..1) |
| `SIGNATURE_ROI_Y_START` | `0.40` | Zone candidate de signature — début Y relatif (0..1) |
| `SIGNATURE_ROI_X_END` | `0.98` | Zone candidate de signature — fin X relatif (0..1) |
| `SIGNATURE_ROI_Y_END` | `0.98` | Zone candidate de signature — fin Y relatif (0..1) |
| `SIGNATURE_BBOX_MARGIN` | `0.08` | Marge ajoutée autour de la bounding box détectée (fraction) |
| `SIGNATURE_BOTTOM_EXCLUSION_RATIO` | `0.10` | **Zone de risque MICR** : bande inférieure de la ROI marquée (fraction de hauteur), **pas** physiquement supprimée |
| `SIGNATURE_MIN_COMPONENT_AREA_RATIO` | `0.0005` | Composantes plus petites que cette fraction de l'aire de ROI → rejetées |
| `SIGNATURE_MAX_COMPONENT_AREA_RATIO` | `0.60` | Composante couvrant plus que cette fraction de l'aire de ROI → rejetée |
| `SIGNATURE_MIN_COMPONENT_WIDTH_RATIO` | `0.01` | Composantes plus fines que cette fraction de la largeur de ROI → rejetées |
| `SIGNATURE_MIN_COMPONENT_HEIGHT_RATIO` | `0.01` | Composantes plus fines que cette fraction de la hauteur de ROI → rejetées |
| `SIGNATURE_MICR_MAX_HEIGHT_RATIO` | `0.06` | Composante « courte » si hauteur < cette fraction de la hauteur de ROI (critère MICR, resserré en V2.2 pour ne pas confondre un long trait diagonal avec un caractère imprimé) |
| `SIGNATURE_MICR_MIN_ASPECT_RATIO` | `6.0` | Composante « très horizontale » si largeur/hauteur > cette valeur (critère MICR) |
| `SIGNATURE_MICR_MIN_WIDTH_RATIO` | `0.03` | Largeur minimale (fraction de la largeur de ROI) pour le critère « court et large » MICR |
| `SIGNATURE_COMPONENT_MERGE_DISTANCE_RATIO` | `0.10` | Regroupement spatial : gaps horizontal/vertical (fractions largeur/hauteur ROI) autorisés entre composantes d'un même groupe (resserré en V2.2 pour ne pas fusionner texte imprimé et signature) |
| `SIGNATURE_GROUP_MIN_COMPONENTS` | `1` | Nombre minimal de composantes pour qu'un groupe soit éligible comme groupe signature principal |
| `SIGNATURE_MORPH_KERNEL_SIZE` | `3` | Kernel du close morphologique (entier impair 1..9), volontairement petit |
| `SIGNATURE_CORE_INK_DISTANCE_RATIO` | `0.018` | **V2.3** : distance d'encre max (fraction de la largeur de ROI, Chebyshev) entre la composante dominante retenue et une composante candidate pour qu'elle rejoigne la structure raffinée |
| `SIGNATURE_CORE_MAX_H_GAP_RATIO` | `0.15` | **V2.3** : gap horizontal max (fraction de la largeur de ROI) pour la règle « même composante » (bbox) du raffinement |
| `SIGNATURE_CORE_MAX_V_GAP_RATIO` | `0.15` | **V2.3** : gap vertical max (fraction de la hauteur de ROI) pour la règle « même composante » (bbox) du raffinement |
| `SIGNATURE_CORE_MIN_HEIGHT_RATIO` | `0.08` | **V2.3** : hauteur minimale (fraction de la hauteur de ROI) d'une composante candidate pour rejoindre la structure raffinée (rejette texte imprimé / specks) |
| `SIGNATURE_AI_ENABLED` | `true` | **AI V2** : active/désactive le service de comparaison IA (métrique apprise). Si désactivé (ou sans PyTorch), l'application démarre normalement, l'OpenCV fonctionne, et `/api/signatures/compare-ai` répond en 503 contrôlé |
| `SIGNATURE_AI_CHECKPOINT` | `ai/checkpoints/metric_resnet18_v2.pt` | **AI V2** : chemin du checkpoint (relatif au dossier racine du service, ou absolu). La métadonnée est validée au chargement (`model_version=ai_metric_v2`, `backbone=resnet18`, `embedding_dim=128`, `canvas_size 256×128`) |
| `SIGNATURE_AI_DEVICE` | `auto` | **AI V2** : `auto` (CUDA si disponible, sinon CPU), `cuda` (erreur contrôlée si indisponible) ou `cpu` |
| `OCR_WORKER_URL` | `http://127.0.0.1:8010` | **OCR Docker V1** : URL du worker PaddleOCR Linux (proxy HTTP depuis le FastAPI Windows) |
| `OCR_WORKER_TIMEOUT` | `60` | **OCR Docker V1** : timeout (s) de l'appel worker (5..300) |

> **PaddleOCR Docker V1 :** `paddle/base/libpaddle.pyd` Windows est bloqué par Smart App Control (`DLL load failed while importing libpaddle`). Le runtime Paddle est isolé dans `ocr-worker/` (Linux Docker, Python 3.11, paddlepaddle 3.2.2 CPU, paddleocr 3.7.0, `127.0.0.1:8010`, volume `ocr-model-cache:/home/worker/.paddlex`). ASP.NET → `POST /api/cheques/ocr` (main FastAPI Windows) → `POST http://127.0.0.1:8010/ocr` (worker) → réponse normalisée. Aucune sécurité Windows modifiée. Voir `ocr-worker/ARCHITECTURE.md` et `ocr-worker/README_LEGACY.md`.


> **Zone candidate (ROI) :** valeurs relatives (0..1) de l'image. La stratégie V1
> suppose que la signature se trouve dans la zone bas-droite d'un chèque standard.
> Ces valeurs ne sont **pas universelles** : elles doivent être calibrées sur les
> vrais formats de chèques utilisés pendant le stage. Contrainte : `0 <= start < end <= 1`
> sur chaque axe (validée au démarrage).

> **Zone analysée :** en V2.1/V2.2, la ROI **entière** est analysée. La bande
> inférieure configurable (`SIGNATURE_BOTTOM_EXCLUSION_RATIO`, nom historique
> conservé) est une **zone de risque MICR** : elle n'est plus retirée avant
> analyse. Seules les composantes qui ressemblent à du MICR / texte imprimé
> **à l'intérieur** de cette bande sont rejetées ; les traits de signature qui
> y descendent sont conservés. Tous les seuils de filtrage des composantes sont
> **relatifs** à la ROI, jamais des coordonnées pixel fixes.

## Démarrage

```bash
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

### GET /api/health

```json
{
  "status": "ok",
  "service": "verification-api",
  "version": "0.1.0"
}
```

### POST /api/images/analyze

Analyse technique d'une image de chèque (décodage OpenCV, dimensions, canaux,
conversion niveaux de gris). Aucun résultat métier.

**Type :** `multipart/form-data`, champ `file`.

**Formats acceptés :** `image/jpeg`, `image/png`.

**Taille maximale :** 10 Mo.

**Codes d'erreur :**
- `400` : fichier vide ou image impossible à décoder
- `413` : fichier trop volumineux
- `415` : type MIME non supporté
- `500` : erreur interne

**Exemple PowerShell :**

```powershell
curl.exe -X POST http://localhost:8000/api/images/analyze `
  -F "file=@C:\path\to\cheque.png;type=image/png"
```

**Exemple curl :**

```bash
curl -X POST http://localhost:8000/api/images/analyze \
  -F "file=@cheque.png;type=image/png"
```

**Réponse (200) :**

```json
{
  "success": true,
  "width": 1600,
  "height": 800,
  "channels": 3,
  "content_type": "image/png",
  "processing": {
    "decoded": true,
    "grayscale_ready": true
  }
}
```

### POST /api/signatures/extract

Extraction **expérimentale** d'une zone de signature à partir d'une image de
chèque (stratégie OpenCV **V2.3**). Ne reconnaît pas la signature, ne calcule aucun
score d'authenticité et ne compare rien.

**Pipeline V2.3 (V2.2 + raffinement dominant-core) :**
1. grayscale + `GaussianBlur(3×3)` ;
2. binarisation **Otsu + `THRESH_BINARY_INV`** → l'encre (sombre) devient le
   premier plan (255) ;
3. analyse de la **ROI entière** — la bande inférieure
   (`SIGNATURE_BOTTOM_EXCLUSION_RATIO`) est marquée **zone de risque MICR**
   mais **pas** retirée : les traits de signature qui y descendent restent
   analysables ;
4. `close` morphologique **petit** et configurable (reconnecte les traits d'un
   même tracé sans fusionner signature + texte + MICR) ;
5. analyse en composantes connexes (`connectedComponentsWithStats`) sur toute
   la ROI ;
6. filtrage **relatif** des composantes (aire min/max, largeur/hauteur min) ;
7. rejet MICR géométrique : dans la zone de risque uniquement, une composante
   est rejetée si elle est **très horizontale** (`largeur/hauteur >
   SIGNATURE_MICR_MIN_ASPECT_RATIO`) ou **courte et large** (hauteur <
   SIGNATURE_MICR_MAX_HEIGHT_RATIO de la ROI **et** largeur ≥
   SIGNATURE_MICR_MIN_WIDTH_RATIO de la ROI). Le seuil « court » est resserré
   (0.06, vs 0.12 en V2.1) : une vraie ligne MICR ne fait qu'environ 4 % de la
   hauteur de la ROI, alors qu'un long trait diagonal qui descend dans la bande
   est bien plus haut — il n'est **jamais** rejeté ;
8. **regroupement spatial** : une composante rejoint un groupe si les gaps
   horizontal **et** vertical avec la composante **la plus proche** du groupe
   sont ≤ `SIGNATURE_COMPONENT_MERGE_DISTANCE_RATIO` (fraction de la
   largeur/hauteur de la ROI). La distance au composant le plus proche (au lieu
   de la bbox du groupe) évite les « sauts » qui fusionnent des grappes
   distinctes ;
9. **sélection du groupe signature** par un score de ressemblance **relatif**
   (et non plus par le nombre de composantes — un texte imprimé contient
   beaucoup de petits caractères) :

   ```
   score = (encre / aire_ROI) × (hauteur_bbox / hauteur_ROI) × (encre / aire_bbox)
   ```

   soit : masse d'encre relative × étendue verticale relative × densité
   (compacité). Les groupes dont ≥ 80 % des composantes sont dans la zone de
   risque MICR sont fortement pénalisés (repli sur le meilleur score si tout
   est dans la bande) ;
10. **raffinement V2.3** : à l'intérieur du groupe sélectionné, la composante
    au plus grand score « dominant-core » devient le **noyau** de la signature ;
    la structure raffinée s'étend ensuite **conservativement** :
    - une composante candidate rejoint la structure si son **encre réelle**
      (masque du composant connexe, pas sa bbox) se trouve à une distance de
      Chebyshev ≤ `SIGNATURE_CORE_INK_DISTANCE_RATIO × largeur_ROI` de l'encre
      d'une composante **déjà acceptée**, ET
    - sa bbox respecte la règle « même composante » (gaps H et V ≤
      `SIGNATURE_CORE_MAX_H_GAP_RATIO` / `SIGNATURE_CORE_MAX_V_GAP_RATIO` de la
      ROI) vis-à-vis d'au moins une composante acceptée, ET
    - sa hauteur ≥ `SIGNATURE_CORE_MIN_HEIGHT_RATIO × hauteur_ROI` (texte
      imprimé / specks sont courts) ;
    - la transformée de distance est **recalculée après chaque acceptation** :
      une grappe imprimée ne peut pas « chaîner » vers la signature via des
      composantes intermédiaires non liées, et le « min-H venant d'une
      composante + min-V venant d'une autre » ne se combine jamais ;
11. bbox finale = **union des composantes raffinées** uniquement (texte imprimé,
    date, « (28) », bruit distant, MICR et composantes d'encre lointaine ne
    l'élargissent pas) ;
12. marge + crop final.

**`extraction_quality` (V2.1/V2.2/V2.3)** — heuristique technique définie ainsi :

```
quality = 0.30 * component_score + 0.40 * density_score + 0.30 * focus_score
component_score = min(1, n_composantes_de_la_structure_raffinée / 3)
density_score   = min(1, densité_d_encre_dans_bbox * 8)
focus_score     = max(0, 1 - bbox_area / roi_area)
```

En **V2.3**, `component_score` et la bbox sont calculés sur la **structure
raffinée** (dominant-core), pas sur le groupe V2.2 complet : le score reflète la
signature isolée, une fois le texte imprimé et l'encre lointaine écartés.

- `component_score` récompense un **groupe cohérent** de plusieurs traits ;
- `density_score` récompense une bbox raisonnablement remplie d'encre et ne
  récompense jamais une bbox vide ;
- `focus_score` pénalise une bbox qui couvre presque toute la ROI (mauvaise
  isolation) **sans** la ramener à 0 : une signature légitime mais large
  conserve une valeur utile et non nulle.

Documentée comme une **heuristique technique** : ce n'est PAS une probabilité,
une authenticité, une conformité ou un score de comparaison. Non persistée.

**Formats acceptés :** `image/jpeg`, `image/png` — **Taille maximale :** 10 Mo.

**Codes d'erreur :**
- `400` : fichier vide ou image impossible à décoder
- `413` : fichier trop volumineux
- `415` : type MIME non supporté
- `422` : ROI invalide, image trop petite, aucun contenu détecté dans la ROI
- `500` : erreur interne

**Exemple PowerShell :**

```powershell
curl.exe -X POST http://localhost:8000/api/signatures/extract `
  -F "file=@C:\path\to\cheque.png;type=image/png"
```

**Réponse (200) :**

```json
{
  "success": true,
  "original_width": 800,
  "original_height": 355,
  "candidate_roi": { "x": 400, "y": 195, "width": 384, "height": 142 },
  "signature_bbox": { "x": 128, "y": 30, "width": 185, "height": 91 },
  "extraction_quality": 0.6575,
  "image_format": "png",
  "signature_image_base64": "..."
}
```

**`signature_image_base64` :** crop final encodé en PNG puis Base64. Choix
accepté uniquement pour le prototype/test ; devra être remplacé (ex. stockage
de fichiers) ultérieurement.

**Attention :** cette extraction repose sur une zone candidate configurable
(voir `SIGNATURE_ROI_*`) et doit être validée sur les formats réels de chèques.

### POST /api/signatures/debug

Endpoint **development-only** de diagnostic de la ROI (utilisé pour calibrer la
zone de recherche de la signature). Il n'est **pas enregistré** hors de
l'environnement `development` (`APP_ENV`) : toute requête reçoit alors un **404**.

**Type :** `multipart/form-data`, champ `file`. Mêmes contraintes que
`/api/signatures/extract` (JPEG/PNG, 10 Mo max).

Il réutilise **exactement** le pipeline d'extraction (mêmes fonctions internes)
et dessine, sur une **copie** de l'image (l'originale n'est jamais modifiée) :
- le rectangle de la **ROI candidate** (vert) ;
- la **zone de risque MICR** (magenta translucide, = bande inférieure
  configurée par `SIGNATURE_BOTTOM_EXCLUSION_RATIO`, désormais **non retirée**) ;
- la **bbox finale** (rouge) si elle est disponible.

Il retourne en plus, pour le diagnostic visuel :
- `mask_image_base64` : le **masque binaire** de la ROI (255 = encre) ;
- `components_image_base64` : la ROI avec les **composantes retenues** dessinées
  (bleu) ;
- `components_rejected_base64` : la ROI avec les composantes rejetées dessinées
  (gris) — **orange** pour celles rejetées comme MICR ;
- `components_all_base64` : la ROI avec **toutes** les composantes (bleu =
  retenues, gris = rejetées, orange = MICR) + la bbox rouge ;
- `group_image_base64` : la ROI avec le **groupe signature principal** + la
  bbox rouge ;
- `micr_band` : coordonnées absolues (sur l'image d'origine) de la zone de
  risque MICR ;

Et les compteurs : `total_component_count`, `retained_component_count`,
`rejected_component_count`, `micr_rejected_count`, `group_count`.

**Diagnostic V2.2 (sélection de groupe) :**
- `groups_image_base64` : ROI avec **tous les groupes générés**, chacun dans
  une couleur distincte + bbox rouge ;
- `groups` : liste d'infos par groupe (`index`, `component_count`, `bbox`,
  `ink_area`, `score` de ressemblance, `selected`) ;
- `selected_group_index` : index du groupe choisi comme candidat signature
  (-1 si aucun) ;
- `selection_reason` : texte expliquant pourquoi ce groupe a été retenu
  (ex. « Groupe 2 sélectionné : score 0.059 (encre relative 0.13, étendue
  verticale 0.46, densité 0.42) ») ;
- `rejected_component_reasons` : nombre de composantes rejetées par motif
  (`aire_trop_petite`, `aire_trop_grande`, `trop_fin`, `micr_texte_imprime`).

**Diagnostic V2.3 (raffinement dominant-core) :**
- `refined_group_image_base64` : ROI avec la **structure raffinée** dessinée
  (noyau dominant en **jaune**, composantes retenues en bleu, composantes du
  groupe sélectionné rejetées en gris) + bbox raffinée rouge ;
- `dominant_component_bbox` : bbox du **noyau dominant** (ROI-local) ;
- `dominant_component_ink` : pixels d'encre **réels** du noyau (masque) ;
- `dominant_component_score` : score dominant-core
  (encre relative × étendue verticale relative × densité) ;
- `refined_component_count` : nombre de composantes de la structure raffinée ;
- `core_component_indices` : indices (dans le groupe sélectionné) **retenus** ;
- `discarded_from_selected_group_indices` : indices (dans le groupe
  sélectionné) **rejetés** par le raffinement ;
- `refined_signature_bbox` : bbox raffinée (== `signature_bbox` en V2.3) ;
- `refinement_reason` : texte expliquant le raffinement (noyau, encre, seuils).

**Réponse (200) :**

```json
{
  "success": true,
  "original_width": 800,
  "original_height": 355,
  "candidate_roi": { "x": 320, "y": 142, "width": 464, "height": 206 },
  "micr_band": { "x": 320, "y": 323, "width": 464, "height": 21 },
  "signature_bbox": { "x": 198, "y": 102, "width": 180, "height": 95 },
  "total_component_count": 42,
  "retained_component_count": 42,
  "rejected_component_count": 6,
  "micr_rejected_count": 6,
  "group_count": 2,
  "groups": [
    { "index": 0, "component_count": 40, "bbox": { "x": 8, "y": 8, "width": 175, "height": 52 }, "ink_area": 5200, "score": 0.0079, "selected": false },
    { "index": 1, "component_count": 2, "bbox": { "x": 198, "y": 102, "width": 180, "height": 95 }, "ink_area": 12985, "score": 0.0476, "selected": true }
  ],
  "selected_group_index": 1,
  "selection_reason": "Groupe 1 sélectionné : score de ressemblance signature 0.04757 (...)",
  "rejected_component_reasons": { "micr_texte_imprime": 6 },
  "dominant_component_bbox": { "x": 198, "y": 102, "width": 180, "height": 95 },
  "dominant_component_ink": 12985,
  "dominant_component_score": 0.0476,
  "refined_component_count": 2,
  "core_component_indices": [0, 1],
  "discarded_from_selected_group_indices": [],
  "refined_signature_bbox": { "x": 198, "y": 102, "width": 180, "height": 95 },
  "refinement_reason": "Raffinement V2.3 : composante dominante (...)",
  "extraction_quality": 0.6575,
  "message": null,
  "image_format": "png",
  "original_with_roi_base64": "...",
  "roi_image_base64": "...",
  "mask_image_base64": "...",
  "components_all_base64": "...",
  "components_rejected_base64": "...",
  "groups_image_base64": "...",
  "group_image_base64": "...",
  "refined_group_image_base64": "...",
  "components_image_base64": "...",
  "signature_image_base64": "..."
}
```

- `original_with_roi_base64` : image originale annotée (ROI verte + zone de
  risque MICR magenta).
- `roi_image_base64` : zone candidate (ROI) annotée (bbox si disponible).
- `signature_image_base64` : crop final (absent si aucun contenu détecté).

**Cas "aucun contenu exploitable" :** pas de crash — le endpoint répond en 200
avec l'image annotée, la ROI, le masque binaire, `signature_bbox`/
`signature_image_base64` à `null`, `extraction_quality` à `0.0` et un `message`
explicite.

Ce diagnostic **ne constitue pas une vérification de conformité**.

### POST /api/signatures/compare

Comparaison expérimentale V1 de deux signatures : la **signature extraite**
(`extracted_file`) et une **signature de référence** (`reference_file`). Les
deux images passent par **exactement le même pipeline de normalisation** avant
le calcul du score.

**Type :** `multipart/form-data`, champs `extracted_file` et `reference_file`.
Mêmes contraintes que `/api/signatures/extract` (JPEG/PNG, 10 Mo max par fichier).

**Pipeline de normalisation (identique pour les 2 signatures) :**
1. décodage OpenCV ;
2. niveaux de gris + léger flou ;
3. binarisation **Otsu + `THRESH_BINARY_INV`** (encre = premier plan), cohérente
   avec la pipeline d'extraction ;
4. rejet des images blanches/sans encre exploitable ;
5. rognage sur la bbox d'encre (suppression des marges) ;
6. redimensionnement **en conservant le ratio** (aucune déformation) ;
7. recentrage sur un canevas fixe (défaut **256×128**, configurable via
   `SIGNATURE_COMPARISON_CANVAS_WIDTH` / `SIGNATURE_COMPARISON_CANVAS_HEIGHT`).

**Métriques (toutes normalisées dans [0, 1]) :**
- `mask_overlap` : IoU des masques d'encre, maximisé sur les petites translations
  (± 4 px) → tolère les petits décalages ;
- `normalized_correlation` : corrélation normalisée moyenne-soustraite
  (`cv2.TM_CCOEFF_NORMED`) → structure des traits ;
- `density_similarity` : `1 - |densité_a - densité_b|` → robuste à l'épaisseur
  de trait / l'échelle.

**Score final :** `0.50 * overlap + 0.35 * corrélation + 0.15 * densité`
(pondérations documentées : overlap = métrique principale, corrélation =
secondaire, densité = la plus faible pour ne pas dominer).

**Réponse (200) :**

```json
{
  "success": true,
  "similarity_score": 0.871234,
  "method": "opencv_baseline",
  "version": "v1",
  "metrics": {
    "mask_overlap": 0.9,
    "normalized_correlation": 0.85,
    "density_similarity": 0.95
  }
}
```

**Codes d'erreur :** `400` (fichier vide / image indécodable), `415` (type MIME
non supporté), `413` (fichier > 10 Mo), `422` (image blanche ou sans encre
exploitable).

**IMPORTANT — limites du score :** le score est une **mesure technique
expérimentale** de similarité visuelle. Ce n'est **pas** une probabilité
d'authenticité, une probabilité de fraude, une décision bancaire ni un verdict
de conformité. **Aucun seuil de décision n'est appliqué** dans ce service : la
conformité reste un traitement ultérieur du workflow.

### POST /api/signatures/compare-ai

Comparaison **IA V2** de deux signatures via le modèle métrique
`siamese_resnet18` (`ai_metric_v2`). Même contrat HTTP que
`/api/signatures/compare` : `multipart/form-data`, champs `extracted_file` et
`reference_file` (JPEG/PNG, 10 Mo max par fichier). Endpoint **additif** — le
baseline OpenCV (`opencv_baseline`, `v1`) est inchangé.

**Prétraitement (parité stricte avec l'évaluation du modèle, aucune
augmentation) :**
1. décodage OpenCV + niveaux de gris ;
2. `GaussianBlur(3×3)` ;
3. binarisation **Otsu + `THRESH_BINARY_INV`** (encre = premier plan) ;
4. rognage sur la bbox d'encre (suppression des marges) ;
5. mise à l'échelle **en conservant le ratio** (jamais d'étirement ni
   d'upscale), recentrage sur un canevas **256×128** (L×H) ;
6. broadcast 3 canaux + normalisation ImageNet.

**Score :** similarité cosinus brute de l'embedding 128-D, dans **[-1, 1]**.
Aucun seuil, aucune probabilité, aucune décision. `method="ai_metric"`,
`version="v2"`, `model="siamese_resnet18"`, `embedding_dimension=128`.

**Réponse (200) :**

```json
{
  "success": true,
  "similarity_score": 0.838154,
  "method": "ai_metric",
  "version": "v2",
  "model": "siamese_resnet18",
  "embedding_dimension": 128,
  "device": "cuda",
  "message": "Comparaison IA effectuée.",
  "extracted_ink_bbox": null,
  "reference_ink_bbox": null
}
```

**Codes d'erreur :** `400` (fichier vide / image indécodable / image sans encre),
`415` (type MIME non supporté), `413` (fichier > 10 Mo), **`503`** (service IA
désactivé, checkpoint invalide ou PyTorch absent) — la réponse 503 reste
structurée (`success=false`, `similarity_score=null`, `device="unavailable"`) et
le baseline OpenCV continue de fonctionner.

**Attention :** le score IA est une **mesure technique** de similarité
d'embedding. Ce n'est pas une probabilité d'authenticité ni un verdict de
conformité ; aucun seuil de décision n'est appliqué ici.

### GET /api/signatures/ai-status

État du service IA : `enabled`, `loaded`, `model`, `version`, `device`.

```json
{ "enabled": true, "loaded": true, "model": "siamese_resnet18", "version": "v2", "device": "cuda" }
```

**Dépendance optionnelle :** PyTorch est requis **uniquement** pour la
comparaison IA (`requirements-ai.txt` ; le checkpoint utilise `ai_metric_v2`).
Sans PyTorch (ou IA désactivée), l'application démarre, l'extraction et la
comparaison OpenCV fonctionnent, et `/compare-ai` répond en 503 contrôlé.

## Calibration de la zone de signature

La zone de recherche de la signature est définie par des **ratios relatifs**
dans la configuration (uniquement côté service, jamais dans le navigateur) :

| Variable | Défaut | Signification |
|---|---|---|
| `SIGNATURE_ROI_X_START` | `0.40` | Début horizontal de la ROI (0..1) |
| `SIGNATURE_ROI_Y_START` | `0.40` | Début vertical de la ROI (0..1) |
| `SIGNATURE_ROI_X_END` | `0.98` | Fin horizontale de la ROI (0..1) |
| `SIGNATURE_ROI_Y_END` | `0.98` | Fin verticale de la ROI (0..1) |
| `SIGNATURE_BBOX_MARGIN` | `0.08` | Marge autour de la bbox détectée (fraction) |
| `SIGNATURE_BOTTOM_EXCLUSION_RATIO` | `0.10` | Zone de risque MICR : bande inférieure de la ROI, **non coupée**, où les composantes ressemblant à du MICR sont rejetées — fraction de la hauteur ROI |
| `SIGNATURE_MIN_COMPONENT_AREA_RATIO` | `0.0005` | Aire minimale d'une composante (fraction de l'aire de ROI) |
| `SIGNATURE_MAX_COMPONENT_AREA_RATIO` | `0.60` | Aire maximale d'une composante (fraction de l'aire de ROI) |
| `SIGNATURE_MIN_COMPONENT_WIDTH_RATIO` | `0.01` | Largeur minimale (fraction de la largeur de ROI) |
| `SIGNATURE_MIN_COMPONENT_HEIGHT_RATIO` | `0.01` | Hauteur minimale (fraction de la hauteur de ROI) |
| `SIGNATURE_MICR_MAX_HEIGHT_RATIO` | `0.06` | Hauteur maximale (fraction de la hauteur de ROI) du critère « court et large » MICR (resserré en V2.2 pour préserver les longs traits diagonaux) |
| `SIGNATURE_MICR_MIN_ASPECT_RATIO` | `6.0` | Seuil largeur/hauteur du critère « très horizontal » MICR |
| `SIGNATURE_MICR_MIN_WIDTH_RATIO` | `0.03` | Largeur minimale (fraction de la largeur de ROI) du critère « court et large » MICR |
| `SIGNATURE_COMPONENT_MERGE_DISTANCE_RATIO` | `0.10` | Regroupement spatial : gaps max (fractions largeur/hauteur ROI) entre composantes d'un même groupe (resserré en V2.2) |
| `SIGNATURE_GROUP_MIN_COMPONENTS` | `1` | Nombre minimal de composantes pour un groupe signature |
| `SIGNATURE_MORPH_KERNEL_SIZE` | `3` | Kernel du close (impair 1..9), volontairement petit |
| `SIGNATURE_CORE_INK_DISTANCE_RATIO` | `0.018` | V2.3 : distance d'encre max (fraction de la largeur de ROI, Chebyshev) du raffinement dominant-core |
| `SIGNATURE_CORE_MAX_H_GAP_RATIO` | `0.15` | V2.3 : gap horizontal max (fraction de la largeur de ROI) de la règle « même composante » |
| `SIGNATURE_CORE_MAX_V_GAP_RATIO` | `0.15` | V2.3 : gap vertical max (fraction de la hauteur de ROI) de la règle « même composante » |
| `SIGNATURE_CORE_MIN_HEIGHT_RATIO` | `0.08` | V2.3 : hauteur minimale (fraction de la hauteur de ROI) d'une composante candidate au raffinement |
| `SIGNATURE_COMPARISON_CANVAS_WIDTH` | `256` | Largeur du canevas de normalisation de la comparaison (32..1024) |
| `SIGNATURE_COMPARISON_CANVAS_HEIGHT` | `128` | Hauteur du canevas de normalisation de la comparaison (32..1024) |

**Procédure recommandée :**
1. Lancer le service en `development` (`APP_ENV=development`).
2. Depuis `Verifications/Create`, sélectionner un chèque puis cliquer sur
   **"Afficher le diagnostic d'extraction"**.
3. Comparer le rectangle vert (ROI) au positionnement réel de la signature sur
   le modèle de chèque utilisé.
4. Ajuster `SIGNATURE_ROI_*` dans `.env` jusqu'à ce que la ROI encadre la
   signature sur plusieurs échantillons, puis redémarrer le service.
5. Le rectangle rouge (bbox) permet de contrôler la précision du crop final.
6. Si la ligne MICR est présente dans la ROI, vérifier sur l'image annotée que
   ses composantes apparaissent **orange** dans `components_all_base64` /
   `components_rejected_base64` (rejetées comme MICR) et **ne s'étendent pas**
   au-delà de la zone de risque magenta. Ajuster
   `SIGNATURE_BOTTOM_EXCLUSION_RATIO`, `SIGNATURE_MICR_*` ou
   `SIGNATURE_COMPONENT_MERGE_DISTANCE_RATIO` si un texte imprimé est retenu ou
   si des traits de signature sont rejetés.
7. Le masque binaire (`mask_image_base64`), l'image des composantes
   (`components_image_base64`) et l'image des groupes (`groups_image_base64`)
   permettent de vérifier que le texte imprimé / les motifs / les grilles sont
   bien exclus de la bbox rouge ; `selection_reason` explique le choix du groupe.
8. En V2.3, `refined_group_image_base64` montre le noyau dominant (jaune), les
   composantes retenues (bleu) et celles rejetées par le raffinement (gris) ;
   `refinement_reason` donne le noyau choisi, l'encre, le nombre de composantes
   retenues/rejetées et les seuils appliqués. Si un texte imprimé fusionné au
   groupe reste dans la bbox rouge, abaisser `SIGNATURE_CORE_INK_DISTANCE_RATIO` ;
   si un vrai trait de signature est rejeté, l'augmenter ou abaisser
   `SIGNATURE_CORE_MIN_HEIGHT_RATIO`.

Les valeurs par défaut supposent une signature en **bas à droite** d'un chèque
standard : elles ne sont **pas universelles** et doivent être validées sur les
formats réels du stage.

## Tests

```bash
.venv\Scripts\activate
python tests\test_image_analyze.py
python tests\test_signature_extraction.py
python tests\test_signature_debug.py
python tests\test_signature_extraction_v2.py
python tests\test_signature_extraction_v21.py
python tests\test_signature_extraction_v22.py
python tests\test_signature_extraction_v23.py
python tests\test_signature_comparison.py

# Tests IA V2 (pytest, nécessite torch — voir requirements-ai.txt)
python -m pytest tests\test_ai_signature_verification.py -q
```

Les scripts démarrent uvicorn sur un port éphémère, exécutent les vérifications
puis arrêtent le serveur. Le test d'extraction couvre : image valide avec
contenu (ROI bornée, crop PNG décodable), image sans contenu rejetée, image
invalide, type non supporté, fichier vide. Le test de debug couvre : disponibilité
en `development`, 404 hors `development`, décodage des images Base64, ROI
dessinée sur une copie, cas sans contenu sans crash. Le test V2 couvre : fond
blanc sans signature, signature synthétique, traits séparés (bbox union), specks
parasites filtrés, bande MICR exclue, bbox bornée, crop décodable, coordonnées
bornées, `extraction_quality` ∈ [0,1] et `/debug` development-only. Le test
**V2.1** couvre : trait diagonal descendant dans la bande MICR conservé (bbox
non tronquée, qualité non nulle), MICR seul rejeté (422), signature + MICR
(bbox excluant la bande, `micr_rejected_count` > 0), regroupement spatial des
composantes proches (`group_count`), parasite distant exclu, fond blanc rejeté,
images/counts debug décodables. Le test
**V2.2** couvre : signature complète avec long trait descendant (bbox contient
la signature entière, y compris le trait dans la bande MICR), texte imprimé
proche non sélectionné malgré un plus grand nombre de composantes, MICR dessous
rejeté, « (28) » isolé non fusionné, date/handwriting éloigné exclu, signature
divisée en composantes proches regroupée, bruit distant exclu, bbox
significativement plus petite que la ROI, fond blanc / MICR seul → 422,
déterminisme et champs debug (`groups`, `selected_group_index`,
`selection_reason`, `rejected_component_reasons`, `groups_image_base64`). Le test
**V2.3** couvre : cluster imprimé fusionné au groupe → rejeté par le raffinement
(bbox raffinée < bbox du groupe, indices rejetés exposés), trait détaché proche
retenu, longue diagonale descendante dans la bande MICR préservée, prévention de
chaînage (composant lointain rejeté, pas de jonction transitive), bbox-2D-proche
mais encre lointaine → rejeté (pas de combinaison min-H/min-V), composante
vraiment 2D-proche → fusionnée, bruit distant exclu, « (28) » isolé exclu, bande
MICR basse exclue, déterminisme (extract + debug) et champs debug V2.3
(`refined_group_image_base64`, `dominant_component_*`, `refined_component_count`,
`core_component_indices`, `discarded_from_selected_group_indices`,
`refined_signature_bbox`, `refinement_reason`). Le test de
comparaison couvre : signatures identiques → score très élevé, même signature
redimensionnée / avec padding différent / avec légère translation → score élevé,
signatures clairement différentes → score inférieur, image blanche rejetée,
fichier invalide → 400, MIME non supporté → 415, score toujours ∈ [0,1] et
déterminisme. Les tests **IA V2** (pytest, `test_ai_signature_verification.py`)
couvrent : service désactivé → erreur contrôlée + 503 + santé/statut OK,
démarrage **sans PyTorch** (sous-processus) avec OpenCV intact, chargement du
checkpoint, fichier manquant / corrompu / métadonnée invalide (`model_version`,
`backbone`, `embedding_dim`, `canvas_size`, `model_state`) → 503 contrôlé,
inférence CPU, résolution de device `auto`/`cuda`/`cpu`, prétraitement parité
avec l'évaluation (déterministe, aucune augmentation), forme de l'embedding
(128-D, norme L2 unitaire), similarité ∈ [-1,1], images identiques → score très
élevé, endpoint réel sur CEDAR (genuine), erreurs contrôlées (image invalide /
blanche → 400), baseline OpenCV inchangé, chargement unique (singleton), service
multi-références (`compare_against_references_ai` : un score par référence,
agrégations max/mean/median/top2_mean, z-normalisation uniquement avec K≥3 sur
statistiques inter-références), et absence totale de seuil/décision dans la
réponse.

## Structure

```
verification-api/
├── app/
│   ├── main.py                          # Application factory + CORS
│   ├── api/routes/
│   │   ├── health.py                    # Endpoint de santé
│   │   ├── images.py                    # POST /api/images/analyze
│   │   ├── signatures.py                # POST /api/signatures/extract
│   │   ├── signatures_compare.py        # POST /api/signatures/compare
│   │   ├── signatures_ai.py             # POST /api/signatures/compare-ai + GET /api/signatures/ai-status
│   │   └── signatures_debug.py          # POST /api/signatures/debug (dev-only)
│   ├── core/
│   │   └── config.py                    # Configuration pydantic-settings (+ ROI, comparaison, IA V2, APP_ENV)
│   ├── schemas/
│   │   ├── health.py                    # Schéma de réponse santé
│   │   ├── image_analysis.py            # Schéma de réponse analyse image
│   │   ├── signature_extraction.py      # Schémas BoundingBox / extraction
│   │   ├── signature_debug.py           # Schéma de réponse du diagnostic
│   │   ├── signature_comparison.py      # Schéma de réponse de la comparaison
│   │   └── ai_signature_comparison.py   # Schémas IA V2 (comparaison + statut + bbox d'encre)
│   └── services/
│       ├── __init__.py                  # Pipeline futur (extraction, comparaison)
│       ├── image_processing_service.py  # Traitement OpenCV (décodage, dimensions)
│       ├── signature_extraction_service.py  # Extraction V2.3 de la zone de signature (OpenCV)
│       ├── signature_comparison_service.py  # Comparaison V1 expérimentale (score 0..1)
│       ├── ai_signature_verification_service.py  # IA V2 : embarqué siamese_resnet18 (ai_metric_v2)
│       └── signature_debug_service.py   # Diagnostic dev-only (ROI dessinée)
├── tests/
│   ├── test_image_analyze.py            # Tests d'intégration de l'analyse
│   ├── test_signature_extraction.py     # Tests d'intégration de l'extraction
│   ├── test_signature_extraction_v2.py  # Tests d'intégration de l'extraction V2
│   ├── test_signature_extraction_v21.py # Tests d'intégration de l'extraction V2.1
│   ├── test_signature_extraction_v22.py # Tests d'intégration de l'extraction V2.2
│   ├── test_signature_extraction_v23.py # Tests d'intégration de l'extraction V2.3
│   ├── test_signature_comparison.py     # Tests d'intégration de la comparaison
│   ├── test_signature_debug.py          # Tests d'intégration du diagnostic
│   └── test_ai_signature_verification.py  # Tests IA V2 (pytest : désactivé, sans torch, checkpoint, inférence, endpoints, multi-références)
├── requirements.txt                     # Dépendances de base (sans torch)
├── requirements-ai.txt                  # PyTorch/torchvision/numpy pour l'IA V2 (cu128)
├── .env.example
└── README.md
```