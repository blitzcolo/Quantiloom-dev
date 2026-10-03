# Fusion-ready exports

A rig job produces native physical-camera observations, separate training truth,
a sample manifest, and a v2 export transaction. Consumers need only JSON/EXR
libraries; neither the SDK nor Vulkan is needed to read the result.

```
Quantiloom.exe fusion-export docs/dataset/fusion/example.toml --dry-run
Quantiloom.exe fusion-export docs/dataset/fusion/example.toml
python scripts/fusion_verify.py build/fusion-example/plate.manifest.json --inspect-exr
```

`fusion.scene_config` and `fusion.output_directory` resolve against the job file.
`sample_id` names the package. Each `[[rig.cameras]]` contains an independent
versioned `sensor` config. `rig_to_world` and `camera_to_rig` are row-major rigid
matrices; camera axes are right/down/forward. Seeds depend on rig/camera identity,
not camera order. Absent `[[rig.pairs]]`, cameras map to `reference_camera`.

## Lenses and products

`sensor.optics.projection.model` is `pinhole`, `brown_conrady`, or `fisheye`.
Optional `intrinsics = [fx,fy,cx,cy]` use top-left coordinates and half-pixel
centres. Without them, focal length/pixel pitch determine fx/fy and the principal
point is centred. Brown coefficients are `[k1,k2,p1,p2,k3]`; fisheye uses
`[k1,k2,k3,k4]` with `max_theta_deg < 90`. Coefficients follow the OpenCV convention
with the declared half-pixel coordinate conversion. Fisheye does not use the
perspective cos-fourth vignetting model.

`fusion.rectify = true` adds bilinear undistortion maps, valid masks, derived
images, freshly traced rectified geometry and corresponding mappings. CFA mosaic
products retain their native grid; demosaiced display products can be rectified.
A derived image links to its parent and does not constitute another acquisition.

Measurements, DN, corrected signals, temperature and display remain distinct
signal kinds. The lens-centre mask is inference input; simulation geometry and
optical path records remain training/evaluation truth. Camera Z and normalized
ray distance are different products, both in metres. Instance IDs are UINT EXR;
zero is background, node index plus one is the stable identity within the frozen
scene manifest. Geometry validity marks misses, invalid lens centres, partial
coverage and surfaces requiring optical path truth.

## Optical models

On a material override, explicitly choose `fusion_transport = "thin_sheet"`
or `"solid"`. Legacy materials retain their existing transport. A thin sheet
uses spectral curves when bound, or authored `fusion_sheet_reflectance` and
`fusion_sheet_transmittance`. These are effective directional coefficients,
including interface losses; their sum cannot exceed one. No additional Fresnel
factor is applied. Absorption determines thermal self-emission.

A solid is closed, manifold, consistently wound, non-scattering, homogeneous and
smooth (`roughness = 0`). It uses wavelength-dependent real-index Snell/Fresnel,
a medium stack, and Beer-Lambert absorption over actual geometry. Set
`fusion_absorption_m_inv`, or derive absorption from measured k when available.
The boundary model uses real-index Fresnel even when bulk absorption comes from
complex-index data; it is not an exact electromagnetic absorbing-interface model.
Solid temperature must be uniform and authored; temperature textures and coupled
thermal solves in solids are rejected. Fluorescent optical interfaces are rejected.
See PBRT 4e Dielectric BSDF and Transmittance for the model equations.

`transmission_job.toml`, `thin_transmission_job.toml`,
`curved_transmission_job.toml`, and `nested_transmission_job.toml` are portable
fixtures. Their shared measurement is two wavelengths around 10 micrometres.

## Truth and limits

Path files have a 128-byte little-endian control header, 32-byte ray records, and
80-byte vertex records, described in their JSON product metadata. They record
actual camera-side paths and first-branch direct/reflected/transmitted radiance
components before detector noise/display encoding. These components reconstruct
the recorded path estimator; they are not calibrated causal attribution.
`fusion.max_recorded_rays` defaults to 4096; zero records all rays, subject to GPU
buffer limits. Stride subsampling and finite-depth truncation are explicit, and
unrecorded/truncated contributions are unknown.

Opaque mappings are checked by a new target ray at continuous native coordinates.
Optical correspondences use wavelength-specific branches and bounded multi-seed
forward tracing/Newton refinement. Verified candidates are retained; no solution
found means unresolved, not invisible. `fusion.max_optical_source_paths` defaults
to 64. The candidate set is not guaranteed complete, and geometric reachability
is not target detectability. Neither PSF nor exposure integration has one unique
surface correspondence. Static rig jobs are the supported first version.

The configuration fingerprint identifies a frozen config, not a complete asset
closure. Export integrity, physical model validity and reproducibility are
separate claims; replayable/reproducibility_verified remain false.
