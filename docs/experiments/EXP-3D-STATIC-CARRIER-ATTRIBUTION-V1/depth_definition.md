# Depth definition

Hypersim `depth_meters` is distance along the unit camera ray in meters. The stored array is used unchanged; asset-unit scale applies to camera translations and native position, not to depth_meters. OpenGL camera axes are converted to OpenCV x-right/y-down/z-forward in data preparation. Per-scene published inverse projection and integer pixel centers with half-pixel resize define K.

GT world point = camera origin + normalize(R K^-1 [u,v,1]) * depth. Camera-z is ray distance times camera-ray z and is generally different. The renderer uses normalized rays and metric samples t, but returns sum(w*t), with zero contribution from escaped mass. It does not divide by opacity. Comparing this unconditional estimate against GT ray distance retains the frozen metric and exposes opacity/coverage failure; no hidden rescaling or depth semantics change is made. Native GT position and dataset camera round-trips are checked independently from any model prediction.
