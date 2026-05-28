# DeepLens Alignment Notes

This project should follow the parts of DeepLens that make the original method
credible:

- Image formation happens in a RAW/linear domain, not directly on display RGB.
- ISP-processed RGB datasets must be treated as a proxy signal and unprocessed
  before optics/sensor simulation.
- Optics are fixed at inference after task optimization.
- Sensor noise and ADC are modeled in digital sensor units with black level and
  ISO/gain, not only as generic normalized image noise.
- RAW task heads should receive packed RGGB-style measurements, optionally with
  metadata such as ISO, so CFA/readout design has a real optimization role.
- Pretrained RGB backbones need physically meaningful RGGB-to-RGB adapter
  initialization; otherwise the sensor path starts from an arbitrary channel
  average and can underperform for reasons unrelated to optics.

Implemented in `train_extended.py`:

- `sensor.sensor_model: deeplens_nbit` enables inverse-ISP unprocessing through
  DeepLens `RGBSensor`, optics/exposure/CFA in RAW-linear space, and n-bit
  black-level/ISO-aware noise plus straight-through quantization.
- `sensor.raw_output: rggbi` packs RAW into RGGB plus an ISO metadata channel.
- `camera_design_best.json` exports `sensor_model`, `raw_output`, `black_level`,
  `iso`, learned CFA weights, exposure, noise/ADC, and PSF parameters.

Implemented in `models/segmentation.py`:

- Pretrained RGB model adapters initialize RGGB as `R, mean(G1,G2), B`.
- RGGB+ISO adapters ignore ISO at initialization but can learn to use it during
  fine-tuning.

Extension beyond DeepLens:

- `sensor.tolerance.enabled: true` enables stochastic physics perturbations
  during training without changing the exported nominal camera: PSF coefficient
  tolerances, exposure calibration drift, CFA spectral-response variation,
  read/shot noise model mismatch, and ADC bit-depth jitter.
- `train.tolerance_task_weight` trains the task head on perturbed camera samples.
- `train.tolerance_consistency_weight` penalizes prediction drift between the
  nominal camera and perturbed-camera pass.
- The stress matrices include `*_robust_codesign_*` rows beside nominal
  `*_codesign_*` rows. This lets the paper evaluate whether the proposed method
  improves DeepLens-style nominal co-design under realistic hardware variation.

Recommended claim framing:

The current KITTI-360 experiments use processed RGB as a proxy scene signal.
Report this honestly as a physically constrained differentiable camera-simulation
frontend unless genuine RAW/calibrated hardware data is added. The strongest
claim should come from fixed-camera vs learned-camera results under explicit
low-light/noisy and low-bit hardware constraints.
