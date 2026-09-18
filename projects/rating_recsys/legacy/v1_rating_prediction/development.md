[2026-04-27]
- Tried CNN ReLU on the last concat (z), multi pooling (& attention) => rmse:~0.5
- Problem in fm_v, leading to training negative rating values for the model => (Adopt uniform distribution)
    => Not that noticeable