# Third-party notices

This project bundles a locally trained chess-square classifier and uses open
data to train it. The application performs inference on-device and does not
send camera frames to a server.

## ONNX Runtime for Android

- Component: `com.microsoft.onnxruntime:onnxruntime-android`
- Version: 1.24.3
- License: MIT
- Copyright: Microsoft Corporation
- Source: https://github.com/microsoft/onnxruntime

The MIT licence text is reproduced in the project's `LICENSE` file for the
Expo template; ONNX Runtime's own notice is available at the source link above.

## FENiT dataset

The square classifier's training data includes square crops derived from the
FENiT annotated board photographs.

- Dataset: Annotated Image Dataset for Fenit: Automatic Chessboard Digitization
  and FEN Generation, Version 1
- Authors: Theo Harjanto, Vincent Oei, Dicky Susanto (Binus University, Jakarta)
- Publisher: Mendeley Data
- DOI: 10.17632/wpp9xnbcp6.1
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Source: https://data.mendeley.com/datasets/wpp9xnbcp6/1

Attribution:

> Harjanto, T., Oei, V., & Susanto, D. (2026). Annotated Image Dataset for
> Fenit: Automatic Chessboard Digitization and FEN Generation (Version 1)
> [Data set]. Mendeley Data. https://doi.org/10.17632/wpp9xnbcp6.1

## Zenodo chess piece dataset

The classifier also uses square crops from physical chess sets 1-3; set 4 is
held out for evaluation only.

- Dataset: Chess Piece Dataset for Image Classification, v1.0.0
- Authors: Pablo Tomas Campos, Antonio Berlanga (Universidad Carlos III de Madrid)
- Publisher: Zenodo
- DOI: 10.5281/zenodo.6656212
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Source: https://zenodo.org/records/6656212

Attribution:

> Campos, P. T., & Berlanga, A. (2022). Chess piece dataset for image
> classification (v1.0.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.6656212

## Trained classifier weights

`rgb_square_classifier_v1.onnx` is an adaptation trained from the two CC BY 4.0
datasets above. The weights are distributed under CC BY 4.0 with the
attributions reproduced in this file. No source dataset images are bundled in
the application; only the trained numerical weights are included.

The surrounding application source code remains under the licence declared in
the repository's `LICENSE` file.
